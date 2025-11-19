import json
import glob
import os
from pathlib import Path
from datetime import datetime

def consolidate_transcripts(session_id_pattern="*"):
    """
    Finds all individual transcript files and merges them into a timestamped master file.
    """
    # 1. Define paths
    transcript_dir = Path("data") / "transcripts"
    
    # --- CHANGE: Unique Filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"interview_full_{timestamp}.json"
    output_file = Path("data") / output_filename
    
    # 2. Find all processed JSON files
    # We look for files ending in '_processed.json' created by asr_wrapper.py
    # Windows safe globbing
    search_pattern = str(transcript_dir / f"{session_id_pattern}_processed.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No transcript files found to merge.")
        return None

    master_transcript = []

    print(f"Found {len(files)} transcript files. Merging...")

    # 3. Loop and Merge
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # We assume each file contains one main "transcript" object
                # We can add a 'source_file' key for debugging
                data["_source_file"] = os.path.basename(file_path)
                master_transcript.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 4. Sort by Timestamp (Optional but recommended)
    # Assumes your JSON has a 'timestamp' field (which asr_wrapper adds)
    try:
        master_transcript.sort(key=lambda x: x.get("timestamp", ""))
    except:
        pass # If timestamps are missing, keep file order

    # 5. Save Master File
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(master_transcript, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully created master transcript: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    consolidate_transcripts()