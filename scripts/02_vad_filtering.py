import os
from core.vad import run_vad_filter
from config import BASE_DIR


if __name__ == "__main__":

    for dir in BASE_DIR.iterdir():
        vad_filtered_dir = dir / "vad_filtered"
        audio_path = dir / "raw" / "recording.mp3"
        output_path = vad_filtered_dir / "recording.wav"

        # create the vad_filtered dir
        os.makedirs(vad_filtered_dir, exist_ok=True)

        if audio_path.exists():
            run_vad_filter(audio_path, output_path)
