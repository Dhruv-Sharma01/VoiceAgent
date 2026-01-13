import logging
import datetime
import json
import asyncio
import random
import base64
import os
from dataclasses import dataclass
from typing import Optional

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
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, groq, silero

# --- NEW: Native Telemetry Imports ---
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("orchard-clinic")


# ==================================================
# 1. NATIVE TRACING SETUP (From Documentation)
# ==================================================
def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    """
    Configures LiveKit to send traces directly to Langfuse via OpenTelemetry.
    This replaces manual 'trace = langfuse.trace()' calls.
    """
    # 1. Get Credentials
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key or not host:
        logger.warning("⚠️ Langfuse credentials missing. Tracing disabled.")
        return

    # 2. Create Auth Header
    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    
    # 3. Configure OpenTelemetry Environment
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    os.environ["OTEL_SERVICE_NAME"] = "Orchard Clinic Agent" 

    # 4. Set the Global Tracer
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)
    logger.info("✅ Native Langfuse Tracing Enabled.")


# ==================================================
# DATA & AGENTS
# ==================================================
@dataclass
class PatientData:
    first_name: str = "John"
    last_name: str = "Smith"
    dob: Optional[str] = None           
    patient_status: Optional[str] = None
    symptoms: Optional[str] = None
    duration: Optional[str] = None       
    insurance_provider: Optional[str] = None
    member_id: Optional[str] = None
    selected_doctor: Optional[str] = None
    appointment_time: Optional[str] = None

class IntakeCoordinatorAgent(Agent):
    def __init__(self, job_ctx: JobContext) -> None:
        self.job_ctx = job_ctx
        now = datetime.datetime.now().strftime("%A, %B %d, %Y")
        super().__init__(
            instructions=f"""
            SYSTEM CONTEXT: You are the Intake Coordinator for 'Orchard Clinic'.
            CURRENT DATE: {now}.

            SCOPE ENFORCEMENT:
            - **MENTAL HEALTH ONLY:** Reject "flu", "broken bones", etc.
            - If physical ailment mentioned: Apologize, say you only treat mental health, and CALL `end_call`.

            WORKFLOW:
            1. **STATUS:** New or Returning?
            2. **TRIAGE:** Symptoms? (Reject physical). Follow up: Duration?
            3. **SCHEDULING:** Day/Time? -> `check_availability`.
            4. **REGISTRATION:** First Name, Last Name, DOB.
            5. **PAYMENT:** Insurance vs Self-Pay. -> `verify_insurance`.
            6. **FINALIZE:** `confirm_appointment`.

            SAFETY: If suicide/harm mentioned -> Call `detect_crisis`.
            """,
            tts=deepgram.TTS(model="aura-asteria-en"),
        )
    
    @function_tool
    async def end_call(self, reason: str):
        logger.info(f"⛔ ENDING CALL. Reason: {reason}")
        await self.job_ctx.room.disconnect()
        return "Call ended."

    async def on_enter(self):
        await self.session.say("Hello, thank you for calling Orchard Clinic. How can I help you today?")

    @function_tool
    async def check_availability(self, context: RunContext[PatientData], day_of_week: str, time_of_day: str):
        if "monday" in day_of_week.lower():
            return "Available: 1. Dr. Puckett (Trauma) 9:30 AM. 2. Dr. Lee (Bipolar) 9:00 AM."
        return f"No slots on {day_of_week}. Suggest Monday morning."

    @function_tool
    async def verify_insurance(self, context: RunContext[PatientData], insurance_provider: str, member_id: str):
        return "Status: Active. Co-pay: $30."

    @function_tool
    async def confirm_appointment(self, context: RunContext[PatientData], doctor_name: str, time: str):
        return "Booking ID: APT-111714."

    @function_tool
    async def detect_crisis(self, context: RunContext[PatientData], reason: str):
        logger.warning(f"CRISIS: {reason}")
        return "Transferring to Crisis Specialist." # Simplified for stability


# ==================================================
# ENTRYPOINT
# ==================================================
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.error(f"Failed to load VAD: {e}")

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # 1. SETUP NATIVE TRACING
    # We call this ONCE. No loops, no manual 'trace=' calls.
    setup_langfuse() 

    # 2. STANDARD LIVEKIT SETUP
    vad_instance = ctx.proc.userdata["vad"]
    session = AgentSession[PatientData](
        vad=vad_instance,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        userdata=PatientData(),
    )

    @ctx.room.on("participant_disconnected")
    def on_user_disconnect(participant: rtc.Participant):
        # if session.current_agent: session.current_agent.tts.stop()
        logger.info("User disconnected.")
        # Native tracing automatically handles the session end
        ctx.shutdown()

    # 3. RUN AGENT
    try:
        await session.start(
            agent=IntakeCoordinatorAgent(job_ctx=ctx),
            room=ctx.room,
            room_input_options=RoomInputOptions(),
            room_output_options=RoomOutputOptions(transcription_enabled=True),
        )
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))