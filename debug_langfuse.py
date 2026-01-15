from langfuse import Langfuse
from dotenv import load_dotenv
import os

load_dotenv(".env.local")

def test_trace():
    print("Initializing Langfuse...")
    langfuse = Langfuse()
    
    # print("Creating trace...")
    # try:
    #     trace = langfuse.trace(name="Debug Trace")
    #     print(f"Trace created: {trace.id}")
    # except AttributeError as e:
    #     print(f"FAILED to create trace: {e}")
    #     return

    print("Fetching dataset...")
    try:
        dataset = langfuse.get_dataset("Medical Agent v1")
        item = dataset.items[0]
        print(f"Got item: {item.id}")
        print(f"Item type: {type(item)}")
        print("Testing item.run() context manager...")
        with item.run(
            run_name="Debug Run",
            # metadata={"test": "true"} # Metadata might not be supported here directly?
        ) as trace:
            print(f"Yielded object type: {type(trace)}")
            print(f"Yielded object dir: {dir(trace)}")
            
            # Simulate scoring
            trace.score(name="test_score", value=1)
            print("Score added to trace created by item.run()")
    except Exception as e:
        print(f"FAILED to interact with dataset: {e}")

if __name__ == "__main__":
    test_trace()
