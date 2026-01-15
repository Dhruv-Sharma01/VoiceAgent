import logging
import os
import base64
import asyncio
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    AgentSession,
    Agent, # Import the base Agent class
)
from livekit.plugins import deepgram, groq, silero
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("trace-test")
logger.setLevel(logging.INFO)

# --- 1. NATIVE TRACING SETUP ---
def setup_langfuse():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.error("❌ MISSING LANGFUSE KEYS. Tracing will fail.")
        return

    # Create Auth Header
    auth_str = f"{public_key}:{secret_key}"
    langfuse_auth = base64.b64encode(auth_str.encode()).decode()

    # Set Environment Variables for LiveKit's Internal Telemetry
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    os.environ["OTEL_SERVICE_NAME"] = "Nano-Agent-Test"

    # Configure Provider
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    set_tracer_provider(provider)
    
    logger.info("✅ OpenTelemetry Provider Set.")

# --- 2. THE NANO AGENT (The Brain) ---
class NanoAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a test agent. Say 'Trace test complete' and nothing else.",
            tts=deepgram.TTS(model="aura-asteria-en")
        )

    async def on_enter(self):
        # We manually trigger a greeting to ensure audio flows
        await self.session.say("Hello. I am the tracing test agent. Please say something.")

# --- 3. ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    setup_langfuse()
    
    await ctx.connect()
    logger.info("🔗 Connected to Room.")

    # Wait for a user to join
    logger.info("⏳ Waiting for participant...")
    participant = await ctx.wait_for_participant()
    logger.info(f"👤 Participant joined: {participant.identity}")

    # Initialize the Session (Container)
    vad = ctx.proc.userdata["vad"]
    session = AgentSession(
        vad=vad,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-asteria-en"),
    )

    # Handle Disconnect
    @ctx.room.on("participant_disconnected")
    def on_user_disconnect(p):
        logger.info("👋 User disconnected. Shutting down...")
        ctx.shutdown()

    # START THE AGENT
    # FIX: Use keyword arguments (agent=..., room=...) to avoid positional errors
    await session.start(agent=NanoAgent(), room=ctx.room)

# --- 4. PREWARM ---
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))