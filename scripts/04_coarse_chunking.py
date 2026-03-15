
from config import BASE_DIR
from core.chunking import save_chunks

if __name__ == "__main__":
  for dir in BASE_DIR.iterdir():
    audio_path = dir / "vad_filtered" / "recording.wav"
    transcript_path = dir / "processed_transcript" / "transcript.txt"
    partial_chunks_path = dir / "partial_chunks"

    if not audio_path.exists() or not transcript_path.exists():
      print(f"audio or transcript missing for {dir}")
      continue

    save_chunks(
        audio_path      = audio_path,
        transcript_path = transcript_path,
        out_dir         = partial_chunks_path,
        chunk_minutes   = 60.0,
        whisper_model   = "large-v3-turbo",
        language        = "en",
        probe_sec       = 30.0,
        overlap_words   = 20,
    )