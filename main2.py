import logging
import datetime
import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Optional, Annotated

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool, ChatMessage
from livekit.plugins import deepgram, groq, silero

from langfuse import Langfuse
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("orchard-clinic")

# ==================================================
# LANGFUSE TRACING
# ==================================================
def setup_langfuse_tracing():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning("Langfuse credentials missing.")
        return

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"
    os.environ["OTEL_SERVICE_NAME"] = "Orchard Clinic Agent"

    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(tp)

langfuse_client = Langfuse()

# ==================================================
# DATA
# ==================================================
@dataclass
class PatientData:
    first_name: str = ""
    last_name: str = ""
    dob: Optional[str] = None
    patient_status: Optional[str] = None
    symptoms: Optional[str] = None
    duration: Optional[str] = None
    insurance_provider: Optional[str] = None
    member_id: Optional[str] = None
    selected_doctor: Optional[str] = None
    appointment_time: Optional[str] = None
    location: Optional[str] = None
    in_crisis: bool = False

# ==================================================
# DUMMY DATA
# ==================================================
DOCTOR_DB = {
    "monday": [
        {"name": "Dr. Puckett", "specialty": "Trauma", "time": "9:30 AM"},
        {"name": "Dr. Lee", "specialty": "Bipolar", "time": "9:00 AM"},
    ],
    "tuesday": [
        {"name": "Dr. Shah", "specialty": "Anxiety", "time": "11:00 AM"},
    ],
}

NEARBY_DOCTORS = {
    "delhi": ["Dr. Puckett", "Dr. Shah"],
    "mumbai": ["Dr. Lee"],
    "bangalore": ["Dr. Shah"],
}

# ==================================================
# CRISIS AGENT
# ==================================================
class CrisisSpecialistAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="You are a Crisis Intervention Specialist. Your ONLY goal is to ensure the patient's safety. Be empathetic, calm, and supportive.",
            tts=deepgram.TTS(model="aura-asteria-en"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        await self.session.say("I’m here with you. Are you safe right now?")

# ==================================================
# INTAKE AGENT
# ==================================================
class IntakeCoordinatorAgent(Agent):
    def __init__(self, job_ctx: JobContext) -> None:
        self.job_ctx = job_ctx
        now = datetime.datetime.now().strftime("%A, %B %d, %Y")
        
        # We move the flow logic into the Prompt to avoid conflict with the code
        super().__init__(
            instructions=f"""
SYSTEM CONTEXT: Orchard Clinic Intake.
DATE: {now}

YOUR GOAL: Collect patient information sequentially.

RULES:
1. **ONE QUESTION AT A TIME**: Never ask multiple questions in one turn. Wait for the user to answer before moving to the next field.
2. **SCOPE**: Only treat mental health cases. If physical ailment (flu, broken bone, etc.) is mentioned, politely decline and end the call.
3. **CRISIS**: If the user mentions suicide, self-harm, or killing themselves, IMMEDIATELY call the `detect_crisis` tool. Do not ask further questions.

INTAKE FLOW (Follow this order strictly):
1. Ask if they are a new or returning patient. (Save to patient_status)
2. Ask for their symptoms. (Save to symptoms)
3. Ask how long they have been experiencing this. (Save to duration)
4. Ask which city they are calling from. (Save to location)
5. Only AFTER collecting the above, help them schedule an appointment or check availability.

Always call the `save_patient_details` tool when you receive new information.
""",
            tts=deepgram.TTS(model="aura-asteria-en"),
        )

    async def on_enter(self):
        # Initial greeting triggers the conversation
        await self.session.say("Hello, thank you for calling Orchard Clinic. Are you a new or returning patient?")

    # NOTE: on_message removed. 
    # The Agent base class handles the loop automatically. 
    # Having a manual on_message + LLM causes double responses.

    # ==================================================
    # TOOLS
    # ==================================================
    
    @function_tool
    async def save_patient_details(
        self, 
        context: RunContext[PatientData], 
        patient_status: Annotated[Optional[str], "New or returning patient"] = None,
        symptoms: Annotated[Optional[str], "Description of symptoms"] = None,
        duration: Annotated[Optional[str], "Duration of symptoms"] = None,
        location: Annotated[Optional[str], "City or location"] = None
    ):
        """Call this tool to save patient information as you collect it."""
        if patient_status: context.userdata.patient_status = patient_status
        if symptoms: context.userdata.symptoms = symptoms
        if duration: context.userdata.duration = duration
        if location: context.userdata.location = location.lower()
        return "Details saved."

    @function_tool
    async def end_call(self, reason: str):
        """Ends the call. Use this for physical ailments or when conversation is done."""
        await self.job_ctx.room.disconnect()
        return "Call ended."

    @function_tool
    async def detect_crisis(self, reason: str):
        """
        URGENT: Call this ONLY if the user mentions suicide, self-harm, or immediate life-threatening danger.
        Do NOT call this for general anxiety or new patients.
        """
        # FIX: Use self.chat_ctx instead of self.session.chat_ctx
        return CrisisSpecialistAgent(chat_ctx=self.chat_ctx), "Transferring to crisis specialist."

    @function_tool
    async def check_availability(self, context: RunContext[PatientData], day_of_week: str, time_of_day: str):
        """Check doctor availability for a specific day."""
        day = day_of_week.lower()
        slots = DOCTOR_DB.get(day)

        if not slots:
            next_day = list(DOCTOR_DB.keys())[0]
            alt = DOCTOR_DB[next_day][0]
            return f"No availability on {day_of_week}. Next best option: {next_day.title()} with {alt['name']} at {alt['time']}."

        doc = slots[0]
        return f"Available: {doc['name']} ({doc['specialty']}) at {doc['time']}."

    @function_tool
    async def suggest_nearby_doctors(self, context: RunContext[PatientData]):
        """Suggests doctors based on the user's location."""
        loc = context.userdata.location
        if not loc:
            return "Location unknown. Ask the user for their city."
            
        docs = NEARBY_DOCTORS.get(loc, [])
        if not docs:
            return "No nearby specialists found. Tele-consultation available."
        return f"Nearby doctors: {', '.join(docs)}."

    @function_tool
    async def verify_insurance(self, context: RunContext[PatientData], insurance_provider: str, member_id: str):
        if not member_id or len(member_id) < 6:
            return "Verification failed. Invalid member ID."
        return "Status: Active. Co-pay: $30."

    @function_tool
    async def confirm_appointment(self, context: RunContext[PatientData], doctor_name: str, time: str):
        return "Booking ID: APT-111714."

# ==================================================
# ENTRYPOINT
# ==================================================
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    setup_langfuse_tracing()

    vad_instance = ctx.proc.userdata.get("vad")
    if vad_instance is None:
        logger.warning("VAD not prewarmed. Loading on demand.")
        vad_instance = silero.VAD.load()
        ctx.proc.userdata["vad"] = vad_instance

    session = AgentSession[PatientData](
        vad=vad_instance,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        userdata=PatientData(),
    )

    agent = IntakeCoordinatorAgent(job_ctx=ctx)
    agent.start(ctx.room) # Ensure agent is started with room context if needed by newer SDKs

    # Removed the manual on_message_wrapper. 
    # AgentSession.start(agent) handles the event loop.

    async def run_post_session_evals():
        transcript = ""
        # Accessing chat_ctx directly from the agent if available, or session if exposed
        chat_ctx = agent.chat_ctx 
        if chat_ctx:
            for msg in chat_ctx.messages:
                if msg.content:
                    transcript += f"{msg.role}: {msg.content}\n"

        if not transcript:
            return

        session_id = ctx.room.name
        langfuse_client.score(
            session_id=session_id,
            name="session_completed",
            value=1,
            comment="Session processed successfully",
        )

    ctx.add_shutdown_callback(run_post_session_evals)

    @ctx.room.on("participant_disconnected")
    def on_user_disconnect(participant: rtc.Participant):
        ctx.shutdown()

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
            room_output_options=RoomOutputOptions(transcription_enabled=True),
        )
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))