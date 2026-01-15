import uuid
import logging
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structure-test")

def run_simulation():
    # 1. Initialize EXACTLY as you requested
    langfuse = get_client()

    # 2. Generate a Session ID (The "Phone Call")
    session_id = f"call_{uuid.uuid4().hex[:8]}"
    logger.info(f"📞 Starting Simulation. Session ID: {session_id}")

    # ==================================================
    # TURN 1: GREETING
    # ==================================================
    logger.info("   -> Simulating Turn 1...")

    # We use 'as_type="trace"' to make this a Root entry in the dashboard
    with langfuse.start_as_current_observation(
        as_type="trace",
        name="turn_1_greeting",
        session_id=session_id, # <--- Link to session
        input="User connected"
    ) as trace:
        
        # Simulate LLM Generation inside the trace
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm_response",
            model="gpt-3.5-turbo"
        ) as generation:
            # Logic happens...
            generation.update(output="Hello! How can I help?")
        
        # Close the trace with final output
        trace.update(output="Hello! How can I help?")

    # ==================================================
    # TURN 2: SYMPTOMS
    # ==================================================
    logger.info("   -> Simulating Turn 2...")

    # Second Context Block = Second Trace (Same Session)
    with langfuse.start_as_current_observation(
        as_type="trace",
        name="turn_2_symptoms",
        session_id=session_id, # <--- Link to SAME session
        input="I have the flu."
    ) as trace:
        
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm_response",
            model="gpt-3.5-turbo"
        ) as generation:
            generation.update(output="We only treat mental health.")

        trace.update(output="We only treat mental health.")

    # ==================================================
    # FLUSH
    # ==================================================
    logger.info("⏳ Uploading...")
    langfuse.flush()
    logger.info("✅ Done.")

if __name__ == "__main__":
    run_simulation()