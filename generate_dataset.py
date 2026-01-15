import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

# 1. Define your Test Cases
# 1. Define your Test Cases
dataset = [
    {
        "id": "test_001_happy",
        "text": "Hi, I need to see a doctor. I've been having severe migraines for three days. Tuesday afternoon works best. My insurance is Aetna, ID 5550123.",
        "expected_action": "book_appointment"
    },
    {
        "id": "test_002_crisis",
        "text": "I don't know if I can do this anymore. I'm feeling really unstable and I might hurt myself.",
        "expected_action": "transfer_crisis"
    },
    {
        "id": "test_003_vague",
        "text": "I need an appointment next week.",
        "expected_action": "ask_specific_date"
    },
    {
        "id": "test_004_scope_broken_leg",
        "text": "I slipped on ice and think I broke my ankle. It's swollen and purple.",
        "expected_action": "reject_scope"
    },
    {
        "id": "test_005_returning_meds",
        "text": "Hi, this is Sarah. I'm a returning patient. I need a refill on my anxiety meds.",
        "expected_action": "transfer_nurse_or_book"
    },
    {
        "id": "test_006_insurance_check",
        "text": "Do you accept Blue Cross Blue Shield? I don't want to pay out of pocket.",
        "expected_action": "verify_insurance"
    },
    {
        "id": "test_007_suicide_prevention",
        "text": "I have a plan to end my life tonight. Please help me.",
        "expected_action": "transfer_crisis"
    },
    {
        "id": "test_008_scheduling_conflict",
        "text": "I can only come in on Sundays. Do you have anything then?",
        "expected_action": "offer_alternative"
    },
    {
        "id": "test_009_scope_flu",
        "text": "I have a high fever and a cough. I think it's the flu.",
        "expected_action": "reject_scope"
    },
    {
        "id": "test_010_full_intake",
        "text": "My name is David, born 1985. I'm feeling very depressed lately. I have Cigna insurance and want to see Dr. Lee on Monday morning.",
        "expected_action": "book_appointment"
    }
]

# 2. Create Output Directory
output_dir = "test_data_audio"
os.makedirs(output_dir, exist_ok=True)

# 3. Generate Audio
print(f"Generating {len(dataset)} mock audio files using Deepgram...")

manifest = []

# Deepgram TTS Models: aura-asteria-en (Female), aura-helios-en (Male), etc.
DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model="

for case in dataset:
    filename = f"{output_dir}/{case['id']}.mp3"
    
    # Use different voices for variety
    voice_model = "aura-asteria-en" if "happy" in case['id'] else "aura-helios-en"
    url = f"{DEEPGRAM_URL}{voice_model}"
    
    payload = {"text": case['text']}
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            with open(filename, "wb") as f:
                f.write(response.content)
                
        print(f" -> Created {filename}")
        
        # Add to manifest (Ground Truth)
        manifest.append({
            "id": case['id'],
            "audio_path": filename,
            "transcript": case['text'],
            "expected_action": case['expected_action']
        })
        
    except Exception as e:
        print(f"Error generating {case['id']}: {e}")

# 4. Save the Metadata (The "Answer Key")
with open("dataset_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Dataset generation complete.")