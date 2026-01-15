import json
import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv(".env.local")

def upload_dataset():
    langfuse = Langfuse()
    dataset_name = "Medical Agent v1"

    # 1. Create (or get) the Dataset
    # We try to create it, if it exists we effectively get it (or we can check).
    # Langfuse SDK creates if not exists usually, or we can use create_dataset explicitly.
    print(f"Creating dataset: {dataset_name}...")
    try:
        langfuse.create_dataset(name=dataset_name, description="Synthetic audio test cases for Intake Agent")
    except Exception as e:
        print(f"Note: Dataset might already exist: {e}")

    # 2. Load Manifest
    with open("dataset_manifest.json", "r") as f:
        manifest = json.load(f)

    # 3. Upload Items
    print(f"Uploading {len(manifest)} items...")
    for case in manifest:
        print(f" -> Uploading {case['id']}...")
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "id": case['id'],
                "text": case['transcript'],
                "audio_path": case['audio_path']
            },
            expected_output={
                "action": case['expected_action']
            },
            metadata={
                "source": "synthetic_generation",
                "voice_model": "deepgram" 
            }
        )
    
    print("Upload complete!")

if __name__ == "__main__":
    upload_dataset()
