"""
HEADLESS AGENT SIMULATION
-------------------------
This script simulates the agent's logic *without* connecting to LiveKit.
It is useful for:
1. Verifying Langfuse tracing integration (spans, generations, scores).
2. Testing the conversation loop and state machine logic rapidly.
3. Debugging LLM prompts and tools without audio latency.

Usage:
    python headless_agent.py
"""
import os
import asyncio
import datetime
from dotenv import load_dotenv
from groq import AsyncGroq
from langfuse import get_client

load_dotenv(".env")

# -----------------------------------
# Clients
# -----------------------------------
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

# -----------------------------------
# LLM call
# -----------------------------------
async def call_agent(messages):
    resp = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512
    )
    return resp.choices[0].message.content

# -----------------------------------
# Example multi-turn conversation
# -----------------------------------
CONVERSATION = [
    "Hi, I’ve been feeling very anxious for the last few weeks.",
    "Sometimes I suddenly feel my heart racing and I get scared.",
    "I’m usually free on weekdays after 6 pm.",
    "I live in Andheri, Mumbai.",
    "I would like to see Dr. Lee tomorrow at 7 pm.",
    "I think I will use insurance to pay."
]
def get_system_prompt():
    return """
You are an empathetic Doctor Matching Agent for Orchard Clinic.

Your job is to understand the user's situation and guide them to the right mental health care in a calm, supportive, and professional way.

-------------------------
CORE RULES
-------------------------

1. SCOPE CONTROL
If the input contains details of any illness that is NOT a mental health condition
(e.g., flu, broken bone, fever, stomach pain):
→ Apologize politely.
→ Say you only handle mental health care.
→ End the conversation.

2. SYMPTOM HANDLING
If the input describes mental health symptoms:
→ First respond with empathy.
→ Then ask:
   - Which weekdays they are available
   - What time of day works for them
   - Their current location

3. SCHEDULING FLOW
If the input contains:
- a doctor name
- and a preferred visit time  
→ Ask whether they want to:
   - Self-pay
   - Or use insurance

4. CRISIS MODE
If the input describes:
- sudden panic attack
- intense fear
- feeling unsafe
- need for immediate help  
→ Immediately:
   - Console the user with empathy
   - Tell them help is being arranged
   - Ask for or confirm their location
   - Say emergency support is being contacted

-------------------------
TONE
-------------------------
Always be:
- Calm
- Empathetic
- Non-judgmental
- Clear and structured

Never be robotic. Never be cold.
Your role is to help a human who may be vulnerable.
"""

# -----------------------------------
# One trace per turn
# -----------------------------------
async def run_multiturn_simulation():

    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]

    session_id = f"sim-{int(datetime.datetime.now().timestamp())}"

    for turn_idx, user_text in enumerate(CONVERSATION, start=1):

        # -------- NEW TRACE --------
        with langfuse.start_as_current_observation(
            as_type="span",
            name="conversation-turn",
            input={
                "session_id": session_id,
                "turn": turn_idx,
                "user_message": user_text,
            },
        ) as root_span:

            messages.append({"role": "user", "content": user_text})

            # -------- AGENT GENERATION --------
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="agent-reply",
                model="llama-3.3-70b-versatile",
                input={"messages": messages},
            ) as gen:

                agent_reply = await call_agent(messages)
                gen.update(output=agent_reply)

            messages.append({"role": "assistant", "content": agent_reply})

            # -------- TRACE OUTPUT --------
            root_span.update(
                output={
                    "agent_reply": agent_reply,
                    "turn": turn_idx,
                }
            )

            root_span.update_trace(
                input={
                    "session_id": session_id,
                    "turn": turn_idx,
                    "user": user_text,
                },
                output={
                    "agent": agent_reply,
                }
            )

            print(f"\nTURN {turn_idx}")
            print("User :", user_text)
            print("Agent:", agent_reply)

    langfuse.flush()

# -----------------------------------
# Entrypoint
# -----------------------------------
if __name__ == "__main__":
    asyncio.run(run_multiturn_simulation())
