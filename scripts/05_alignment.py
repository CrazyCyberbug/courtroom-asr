import json
from core.aligner import produce_alignments
from config import BASE_DIR

for dir in BASE_DIR.iterdir():

  partial_chunks_path = dir / "partial_chunks"
  manifest_path = partial_chunks_path / "manifest.json"

  if not partial_chunks_path.exists() or not manifest_path.exists():
    continue

  with open(manifest_path, "r") as f:
    manifest = json.load(f)

  num_chunks = len(manifest)

  for i in range(1, num_chunks+1):
    chunk_path = partial_chunks_path / f"chunk_{i}.wav"
    transcript_path = partial_chunks_path / f"chunk_{i}_transcript.txt"
    word_timestamps_path = partial_chunks_path / f"chunk_{i}_word_timestamps.json"

    if not chunk_path.exists() or not transcript_path.exists():
      print(f"chunk {i} is missing")
      continue

    word_timestamps = produce_alignments(chunk_path, transcript_path, word_timestamps_path)
    with open(word_timestamps_path, "w") as f:
      json.dump(word_timestamps, f)