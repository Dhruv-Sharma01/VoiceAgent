"""
MAIN APPLICATION ENTRYPOINT
---------------------------
This module defines the LiveKit Worker for the Orchard Clinic Intake Agent.

Architecture:
- **Infrastructure**: LiveKit (Real-time Audio/Video/Data)
- **Intelligence**: Groq (Llama 3.3 70B)
- **Speech**: Deepgram (STT/TTS)
- **Observability**: OpenTelemetry (OTLP) -> Langfuse

Features:
- State Machine for Patient Intake (Status -> Symptoms -> Schedule -> Insurance)
- Crisis Detection & Handover
- Tool Calling (Doctor DB, Insurance Verification)
- Automated Post-Session Evaluations
"""
import logging
import datetime
import json
import asyncio
import random
import base64
import os
from dataclasses import dataclass
from typing import Optional, List

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

# --- NATIVE TRACING IMPORTS ---
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("orchard-clinic")

# ==================================================
# 1. NATIVE TRACING SETUP
# ==================================================
def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key or not host:
        logger.warning("⚠️ Langfuse credentials missing. Tracing will be disabled.")
        return

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    os.environ["OTEL_SERVICE_NAME"] = "Orchard Clinic Agent"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)
    logger.info("✅ Native Langfuse Tracing Enabled.")


# ==================================================
# DATA MODELS
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


# ==================================================
# HELPERS
# ==================================================
# ==================================================
# HELPER (Unused)
# ==================================================

# ==================================================
# AGENTS
# ==================================================
class CrisisSpecialistAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="You are a Crisis Intervention Specialist. Your ONLY goal is safety. Ask: 'Are you safe right now?' and wait for a response.",
            tts=deepgram.TTS(model="aura-asteria-en"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.warning("🚑 CRISIS PROTOCOL ACTIVATED")
        # Edge case: Ensure we don't speak over the previous agent immediately
        await asyncio.sleep(0.5) 
        await self.say("I am connecting you to a crisis specialist. Are you safe right now?")

class IntakeCoordinatorAgent(Agent):
    def __init__(self, job_ctx: JobContext) -> None:
        self.job_ctx = job_ctx
        now = datetime.datetime.now().strftime("%A, %B %d, %Y")
        super().__init__(
            instructions=f"""
            SYSTEM CONTEXT: You are the Intake Coordinator for 'Orchard Clinic'.
            CURRENT DATE: {now}.

            === EDGE CASE HANDLING ===
            1. **AMBIGUITY:** If user says "Tomorrow" or "Next week", ask them to clarify the specific Day of the Week (e.g., "Do you mean Monday?").
            2. **REFUSAL:** If user says "I don't have insurance", ask if they want to pay out-of-pocket ($250). If yes, skip insurance steps.
            3. **IRRELEVANCE:** If user asks about weather/sports, politely say "I'm not sure, but let's get you booked." and return to the step.
            4. **SILENCE/MUMBLING:** If you didn't catch that, ask "Could you please repeat that?"
            5. **PHYSICAL AILMENT:** If user mentions "flu", "broken bone", or "blood", apologize and call `end_call`.

            === WORKFLOW (LINEAR) ===
            1. **STATUS:** New or Returning?
            2. **TRIAGE:** Symptoms? (Save to data). Follow up: Duration?
            3. **SCHEDULING:** Ask for Day/Time -> Call `check_availability`. 
               - *NOTE:* If requested slot is taken, offer alternatives.
            4. **REGISTRATION:** First Name, Last Name, DOB.
            5. **PAYMENT:** Insurance (Provider + Member ID) OR Self-Pay.
               - If Insurance -> `verify_insurance`.
            6. **FINALIZE:** Call `confirm_appointment`.

            === SAFETY ===
            If user mentions "suicide", "kill myself", "harm", or "weapon" -> IMMEDIATELY Call `detect_crisis`.
            """,
            tts=deepgram.TTS(model="aura-asteria-en"),
        )

    # --- REQUIRED: Message Loop to keep conversation going ---
    async def on_message(self, msg: ChatMessage):
        # This function ensures the agent actually replies to the user
        if not msg.content: return
        await self.session.generate_reply()

    @function_tool
    async def end_call(self, reason: str):
        logger.info(f"⛔ ENDING CALL. Reason: {reason}")
        await self.job_ctx.room.disconnect()
        return "Call ended."

    async def on_enter(self):
        await self.session.say("Hello, thank you for calling Orchard Clinic. How can I help you today?")

    @function_tool
    async def check_availability(self, context: RunContext[PatientData], day_of_week: str, time_of_day: str):
        # Edge Case: Handle inputs like "tomorrow" or "next monday" by looking for keywords
        query = day_of_week.lower()
        
        if "monday" in query:
            return "Available: 1. Dr. Puckett (Trauma) 9:30 AM. 2. Dr. Lee (Bipolar) 9:00 AM."
        elif "tuesday" in query:
            return "Available: Dr. Shah (Anxiety) 11:00 AM."
        
        # Edge Case: User asks for a weekend or invalid day
        return f"I'm sorry, we don't have slots on {day_of_week}. We have openings this Monday and Tuesday."

    @function_tool
    async def verify_insurance(self, context: RunContext[PatientData], insurance_provider: str, member_id: str):
        # Edge Case: Short/Invalid ID. 
        # Real-world IDs are alphanumeric and > 5 chars.
        if len(member_id) < 5 or not any(c.isdigit() for c in member_id):
            return "The Member ID seems invalid (too short or missing numbers). Please verify."
        return "Status: Active. Co-pay: $30."

    @function_tool
    async def confirm_appointment(self, context: RunContext[PatientData], doctor_name: str, time: str):
        return "Booking ID: APT-111714."

    @function_tool
    async def detect_crisis(self, context: RunContext[PatientData], reason: str):
        logger.warning(f"CRISIS: {reason}")
        # FIX: Use self.chat_ctx (from the agent instance) to avoid AttributeError
        return CrisisSpecialistAgent(chat_ctx=self.chat_ctx), "Transferring."


    @ctx.room.on("participant_disconnected")
    def on_user_disconnect(participant: rtc.Participant):
        if session.current_agent: 
            # Edge Case: Stop audio immediately on disconnect
            try: session.current_agent.tts.stop()
            except: pass
        logger.info("User disconnected.")
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