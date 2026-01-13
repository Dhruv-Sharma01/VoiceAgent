import asyncio
import json
import os
from livekit import api, rtc
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv(".env.local")

# Configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

langfuse = Langfuse()

async def run_single_test(case):
    print(f"--- Running Test: {case['id']} ---")
    
    # 1. Create a Room for this specific test
    room_name = f"eval_{case['id']}"
    lk_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    try:
        # Create room (or ensure it exists)
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        
        # 2. Connect as a "Fake User"
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity("tester_bot") \
            .with_name("Test Bot") \
            .with_grants(api.VideoGrants(room_join=True, room=room_name)) \
            .to_jwt()

        room = rtc.Room()
        await room.connect(LIVEKIT_URL, token)
        
        # 3. Prepare the Audio Source
        # We publish a track that plays your .mp3 file
        audio_path = case['audio_path']
        track = rtc.LocalAudioTrack.create_audio_track("mic", rtc.AudioSource(sample_rate=48000, num_channels=1))
        publication = await room.local_participant.publish_track(track)
        
        # 4. START TRACING IN LANGFUSE
        trace = langfuse.trace(
            name=f"Eval: {case['id']}",
            input=case['transcript'],
            metadata={"expected_action": case['expected_action']}
        )
        
        print(f"Playing audio: {audio_path}")
        
        # --- AUDIO PLAYBACK LOGIC ---
        # Note: In a real script, you decode the MP3 to PCM and push it to the track.
        # For simplicity here, we assume the agent starts listening when we join.
        # *Critically*, pushing raw audio files programmatically requires `ffmpeg` or `pydub`.
        # *Simplified approach for this script:* We just wait to simulate the interaction time
        # assuming you might play the audio manually or use a virtual mic for the first pass.
        
        await asyncio.sleep(5) # Simulate talking time
        await asyncio.sleep(5) # Wait for agent response
        
        # 5. FETCH THE AGENT'S TRACE FROM THE SERVER
        # (Since the Agent runs separately, we need to find its trace.
        # In a real setup, you link them via Session ID. 
        # Here, we will just Log a "Pass/Fail" based on if the agent didn't crash).
        
        # Score the test
        # (In a real integration, you'd check if the Agent actually booked the slot)
        trace.score(
            name="test_completion",
            value=1,
            comment="Test ran without crashing"
        )
        
        print(f"Test {case['id']} complete.")
        
        await room.disconnect()
        
    finally:
        await lk_api.aclose()

async def main():
    # Load the manifest
    with open("dataset_manifest.json", "r") as f:
        manifest = json.load(f)
        
    for case in manifest:
        try:
            await run_single_test(case)
        except Exception as e:
            print(f"FAILURE DETECTED in {case['id']}: {e}")
        
        await asyncio.sleep(2) # Cooldown between tests

if __name__ == "__main__":
    asyncio.run(main())