import json
import torchaudio
from pathlib import Path
from config import BASE_DIR, HF_USERNAME
from core.segment_mapper import build_dataset, push_to_hub


if __name__ == "__main__":  
    df = build_dataset()
    keep_columns = ["audio", "text", "speaker", "case_name", "confidence"]
    push_to_hub(df, keep_columns,HF_USERNAME,"courtroom-asr-dataset")
    