import logging
import datetime
import json
import asyncio
import base64
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

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

# --- NATIVE TRACING ---
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("orchard-clinic")

# ==================================================
# 0. SETUP & GLOBAL MEMORY
# ==================================================
SESSION_DB: Dict[str, dict] = {}
SESSION_DB: Dict[str, dict] = {}
REMOTE_DB: Dict[str, dict] = {}
WAITLIST_DB: List[dict] = []

def setup_langfuse():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if not public_key or not secret_key: return
    
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"
    os.environ["OTEL_SERVICE_NAME"] = "Orchard Clinic Agent"
    
    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(tp)

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
    generated_id: Optional[str] = None

# ==================================================
# 1. THE JUDGE MODULE
# ==================================================
class ConversationJudge:
    """
    Evaluates the conversation history in real-time.
    """
    def __init__(self):
        # We use a fast model for the Judge to minimize latency
        self.llm = groq.LLM(model="llama-3.3-70b-versatile") 

    async def evaluate(self, history: List[ChatMessage]) -> Optional[str]:
        """
        Returns an instruction string if intervention is needed, else None.
        """
        # 1. Prepare Transcript for the Judge
        transcript = "\n".join([f"{m.role.upper()}: {m.content}" for m in history[-6:]]) # Look at last 6 turns
        
        prompt = f"""
        You are a conversation supervisor. Review the transcript below.
        
        TRANSCRIPT:
        {transcript}
        
        CHECK FOR:
        1. **Circles**: Is the agent asking the same question repeatedly?
        2. **Deviation**: Is the user talking about unrelated topics (sports, weather)?
        1. **Circles**: Is the agent asking the same question repeatedly?
        2. **Compound Questions**: Is the agent asking for too much at once? (e.g., Name + DOB + Insurance)?
           - RULE: ONE topic per turn.
        3. **Deviation**: Is the user talking about unrelated topics?
        4. **Hostility**: Is the user angry?
        4. **DATA COMPLETENESS**: 
           - Did the agent skip asking for "First Name", "Last Name", or "DOB"?
           - Did it skip "Member ID" before saying insurance is verified?
           - If YES, INSTRUCT agent to go back and ask for them.
        
        OUTPUT FORMAT:
        - If everything is normal, output: "OK"
        - If there is an issue, output a concise SYSTEM INSTRUCTION for the agent.
          Example: "User is deviating. Politely steer back to insurance details."
        
        YOUR VERDICT (Single line):
        """
        
        # 2. Query Judge LLM
        # We create a temporary context just for this judgment call
        judge_ctx = ChatContext().append(role="user", text=prompt)
        
        try:
            stream = await self.llm.chat(chat_ctx=judge_ctx)
            verdict = ""
            async for chunk in stream:
                verdict += chunk.choices[0].delta.content or ""
            
            verdict = verdict.strip()
            logger.info(f"👨‍⚖️ JUDGE VERDICT: {verdict}")
            
            if "OK" in verdict or len(verdict) < 5:
                return None # No intervention needed
            return verdict
            
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return None

# ==================================================
# DUMMY API
# ==================================================
async def save_to_remote_db(user_identity: str, patient_data: PatientData, chat_history: List[ChatMessage]):
    print(f"\n⚡ [API] Saving session for {user_identity}...")
    await asyncio.sleep(0.1) 
    
    transcript_text = "\n".join([f"{m.role}: {m.content}" for m in chat_history if m.content])
    
    # Simple Append Logic
    if user_identity not in REMOTE_DB:
         REMOTE_DB[user_identity] = []
    
    REMOTE_DB[user_identity].append({
        "date": datetime.datetime.now().isoformat(),
        "transcript": transcript_text,
        "data": asdict(patient_data)
    })
    print(f"✅ [API] Data Saved.")

# ==================================================
# AGENTS
# ==================================================
class CrisisSpecialistAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="You are a Crisis Specialist. Ask: 'Are you safe right now?'",
            tts=deepgram.TTS(model="aura-asteria-en"),
            chat_ctx=chat_ctx,
        )
    async def on_enter(self):
        await asyncio.sleep(0.5) 
        if self.session:
            await self.session.say("I am connecting you to a crisis specialist. Are you safe right now?")

class IntakeCoordinatorAgent(Agent):
    def __init__(self, job_ctx: JobContext, user_identity: str):
        self.job_ctx = job_ctx
        self.user_identity = user_identity
        self.judge = ConversationJudge() # <--- Instantiate the Judge
        
        now = datetime.datetime.now().strftime("%A, %B %d, %Y")
        super().__init__(
            instructions=f"""
            SYSTEM CONTEXT: You are the Intake Coordinator for 'Orchard Clinic'.
            CURRENT DATE: {now}.

            === WORKFLOW ===
            1. **STATUS:** New/Returning?
            2. **TRIAGE:** Symptoms & Duration.
            3. **SCHEDULING:** Day/Time -> `check_availability`.
            4. **REGISTRATION**: 
               - Step A: Ask for "First Name, Last Name, and Date of Birth".
               - Step B: Wait for user to answer.
            5. **PAYMENT METHOD**: 
               - Ask: "Will you be using insurance or paying out of pocket?"
               - If Self-Pay -> Confirm $250 rate -> Move to Finalize.
            6. **INSURANCE DETAILS** (Only if using insurance):
               - Ask for "Insurance Provider and Member ID".
               - `verify_insurance`.
            7. **FINALIZE**: `confirm_appointment`.

            === CONVERSATION STYLE ===
            - **ONE QUESTION AT A TIME**: Do not bundle requests. 
            - Example BAD: "Can I have your name and do you have insurance?"
            - Example GOOD: "Can I have your name?" -> (Wait) -> "Do you have insurance?"

            === CRITICAL: SEMANTIC DISTINCTION ===
            - **Depression/Anxiety/Sadness** -> These are SYMPTOMS we treat. CONTINUE intake.
            - **"I want to die" / "I have a gun" / "I will kill myself"** -> These are EMERGENCIES.
            
            === SAFETY PROTOCOL ===
            ONLY Call `detect_crisis` if the user explicitly threatens IMMEDIATE self-harm or suicide.
            DO NOT call it for general depression or "feeling down".
            """,
            tts=deepgram.TTS(model="aura-asteria-en"),
        )

    # --- SAVE STATE ---
    def save_state(self):
        try:
            # DEBUG: Find where the history lives
            source = None
            if hasattr(self.session, "chat_context"):
                source = self.session.chat_context
            elif hasattr(self.session, "chat_ctx"):
                source = self.session.chat_ctx
            elif hasattr(self, "chat_ctx"):
                source = self.chat_ctx
            
            # Try to get messages
            messages = []
            if source:
                 if hasattr(source, "messages"):
                     messages = source.messages
                 elif isinstance(source, list):
                     messages = source
                 # Special check for _ReadOnlyChatContext which might need a property
                 elif hasattr(source, "to_list"): 
                     messages = source.to_list()
            
            if not messages:
                 # If we still can't find it, LOG IT so we can fix it next turn
                 # logger.warning(f"DEBUG: Could not find messages. Session items: {dir(self.session)}")
                 # logger.warning(f"DEBUG: Agent items: {dir(self)}")
                 return

            history_dump = [{"role": m.role, "content": m.content} for m in messages]
            
            SESSION_DB[self.user_identity] = {
                "history": history_dump,
                "data": self.session.userdata
            }
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # --- INTERCEPT USER MESSAGE ---
    async def on_message(self, msg: ChatMessage):
        if not msg.content: return

        # 1. EVALUATE WITH JUDGE (Before replying)
        # Use session context for history
        current_history = self.session.chat_ctx.messages if hasattr(self.session.chat_ctx, "messages") else list(self.session.chat_ctx)
        advice = await self.judge.evaluate(current_history)
        
        # 2. INJECT JUDGMENT (If any)
        if advice:
            logger.warning(f"💉 INJECTING ADVICE: {advice}")
            # We add a temporary System message that guides the agent for THIS turn only
            # The agent will see this immediately before generating its answer.
            injection = f"[SUPERVISOR INSTRUCTION]: {advice}"
            self.chat_ctx.append(role="system", text=injection)

        # 3. GENERATE REPLY
        await self.session.generate_reply()
        
        # 4. SAVE STATE
        self.save_state()

    # --- TOOLS ---
    @function_tool
    async def end_call(self, reason: str):
        await self.job_ctx.room.disconnect()
        return "Call ended."

    @function_tool
    async def check_availability(self, context: RunContext[PatientData], day_of_week: str, time_of_day: str):
        self.save_state()
        query_day = day_of_week.lower()
        query_time = time_of_day.lower()

        # Logic: Only Monday mornings (AM) are available
        is_monday = "monday" in query_day
        is_morning = "am" in query_time
        is_pm = "pm" in query_time
        
        # 1. Reject PM / Late hours
        if is_pm:
             return f"We are closed at {time_of_day}. Our hours are 9 AM to 5 PM."

        # 2. Offer Slots if valid
        if is_monday and is_morning:
            return "Available: Dr. Puckett (9:30 AM), Dr. Lee (9:00 AM)."
        
        # 3. Waitlist Logic (No slots)
        WAITLIST_DB.append({
            "user_id": self.user_identity,
            "day": day_of_week,
            "time": time_of_day,
            "symptoms": context.userdata.symptoms
        })
        
        return (f"I'm sorry, we don't have any appointments available at {time_of_day} on {day_of_week}. "
                "I have added you to our priority waitlist. We will contact you as soon as a slot opens up.")

    @function_tool
    async def verify_insurance(self, context: RunContext[PatientData], insurance_provider: str, member_id: str):
        self.save_state()
        return "Status: Active. Co-pay: $30."

    @function_tool
    async def confirm_appointment(self, context: RunContext[PatientData], doctor_name: str, time: str):
        self.save_state()
        return "Booking ID: APT-111714."

    @function_tool
    async def detect_crisis(self, context: RunContext[PatientData], reason: str):
        return CrisisSpecialistAgent(chat_ctx=self.chat_ctx), "Transferring."

    async def on_enter(self):
        logger.info(f"DEBUG: chat_ctx type: {type(self.chat_ctx)}")
        logger.info(f"DEBUG: chat_ctx dir: {dir(self.chat_ctx)}")
        
        has_history = False
        try:
            # Attempt to check history safely
            if hasattr(self.chat_ctx, "messages"):
                has_history = len(self.chat_ctx.messages) > 1
            elif hasattr(self.chat_ctx, "__len__"):
                has_history = len(self.chat_ctx) > 1
        except:
            pass

        if has_history:
            await self.session.say("I have retrieved your details. Let's continue.")
        else:
            await self.session.say("Hello, thank you for calling Orchard Clinic. How can I help you today?")

# ==================================================
# ENTRYPOINT
# ==================================================
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    setup_langfuse()

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    user_id = participant.identity
    logger.info(f"👤 Connected: {user_id}")

    # --- RESTORE STATE ---
    initial_data = PatientData()
    initial_history = []
    
    if user_id in SESSION_DB:
        logger.info("found saved session")
        saved = SESSION_DB[user_id]
        initial_data = saved["data"]
        initial_history = saved["history"]

    vad_instance = ctx.proc.userdata["vad"]
    session = AgentSession[PatientData](
        vad=vad_instance,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        userdata=initial_data,
    )
    
    agent = IntakeCoordinatorAgent(job_ctx=ctx, user_identity=user_id)
    
    # Inject History
    if initial_history:
        for m in initial_history:
            if m["role"] != "system": # Don't duplicate old system prompts
                agent.chat_ctx.append(role=m["role"], text=m["content"])
        agent.chat_ctx.append(role="system", text="[SYSTEM]: Session restored. Resume conversation.")

    # Shutdown Hook
    async def shutdown_handler():
        try:
            ctx_source = session.chat_ctx
            final_history = ctx_source.messages if hasattr(ctx_source, "messages") else list(ctx_source)
            await save_to_remote_db(user_id, session.userdata, final_history)
        except Exception as e:
            logger.error(f"Failed to save to remote DB: {e}")

    ctx.add_shutdown_callback(shutdown_handler)

    @ctx.room.on("participant_disconnected")
    def on_user_disconnect(participant: rtc.Participant):
        if session.current_agent: 
             try: session.current_agent.tts.stop()
             except: pass
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