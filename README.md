# Orchard Clinic AI Intake Agent

This repository contains the implementation of an AI-powered Intake Coordinator for Orchard Clinic, designed to handle mental health patient intake calls. The agent is built using **LiveKit** for real-time voice interaction, **Groq (Llama 3.3)** for intelligence, and **Langfuse** for observability and evaluation.

## Key Files

### 1. `main.py`
The core application entry point.
- **Functionality**: Runs the LiveKit worker, handling real-time audio (STT/TTS) and conversation logic.
- **Architecture**: Implements a dedicated State Machine for the intake workflow (Status -> Symptoms -> Duration -> Location -> Schedule -> Insurance).
- **Features**:
  - **Scope Enforcement**: Strictly rejects physical ailments (e.g., "broken leg").
  - **Crisis Protocol**: Detects self-harm keywords and transfers to a `CrisisSpecialistAgent`.
  - **Tools**: Includes function tools for checking doctor availability, verifying insurance, and confirming appointments.
  - **Observability**: configured with native OpenTelemetry tracing to send full call traces to Langfuse.

### Robust Edge Case Handling
The agent includes specific logic to handle real-world conversation challenges:
1. **Ambiguity Resolution**: Clarifies vague dates like "tomorrow" or "next week" (e.g., "Do you mean Monday?").
2. **Refusal Handling**: If insurance is refused, seamlessly pivots to self-pay options ($250).
3. **Irrelevance Filter**: Politely dismisses off-topic queries (weather, sports) and steers back to intake.
4. **Silence/Mumbling**: Asks the user to repeat if speech input is unclear.
5. **Strict Scope Control**: Immediately ends calls for physical ailments (flu, broken bones) to ensure safety.
6. **Crisis Safety Net**: Prioritizes safety above all; transfers to a specialist if self-harm is detected.

### 2. `headless_agent.py`
A testing utility for the agent's logic.
- **Purpose**: Simulates the agent's conversation flow *without* the LiveKit audio infrastructure.
- **Usage**: Useful for rapid logic verification and testing the Langfuse tracing integration logic (using context managers) without needing to connect a voice client.
- **Implementation**: Mocks the conversation loop and directly calls the LLM and tools.

### 3. `headless_agent2.py`
A minimal, multi-turn simulation example.
- **Purpose**: Demonstrates a clean, decorator-less pattern for instrumenting multi-turn conversations with Langfuse.
- **Highlights**: Shows how to create nested spans (`conversation-turn` -> `agent-reply`) for granular observability of the agent's thought process and latency.

## Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env.local` file with:
   ```
   LIVEKIT_URL=...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   GROQ_API_KEY=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_BASE_URL=...
   DEEPGRAM_API_KEY=...
   ```

3. **Run the Agent**:
   ```bash
   python main.py dev
   ```

4. **Run Headless Simulation**:
   ```bash
   python headless_agent.py
   ```
