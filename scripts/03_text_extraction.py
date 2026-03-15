import os
import json
from core.text import preprocess_transcripts
from config import BASE_DIR


for dir in BASE_DIR.iterdir():
  case_raw_dir = dir / "raw"
  if not case_raw_dir.exists():
    continue

  processed_trancript_dir = dir / "processed_transcript"
  os.makedirs(processed_trancript_dir, exist_ok=True)
  
  raw_transcript_path = case_raw_dir / "transcript.pdf"
  if raw_transcript_path.exists():
    text, records, clean_records = preprocess_transcripts(raw_transcript_path)

    with open(processed_trancript_dir / "transcript.txt", "w") as f:
      f.write(text)

    with open(processed_trancript_dir / "records.json", "w") as f:
      json.dump(records, f, indent = 2)

    with open(processed_trancript_dir / "clean_records.json", "w") as f:
      json.dump(clean_records, f, indent = 2)
