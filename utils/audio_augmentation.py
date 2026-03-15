from datasets import concatenate_datasets, load_from_disk
from audiomentations import Compose, AddGaussianNoise, TimeStretch, Gain, Shift
from io import BytesIO
import soundfile as sf
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
    TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.3),
])

def augment_audio(example):
    audio_obj = example["audio"]

    # Defensively cast to numpy — datasets may pass a plain list during map()
    audio = np.array(audio_obj["array"], dtype=np.float32)
    sr = audio_obj["sampling_rate"]

    augmented = augment(samples=audio, sample_rate=sr)
    augmented = np.asarray(augmented, dtype=np.float32)

    # Encode to WAV bytes directly — avoids the cast_storage list/.T bug
    buffer = BytesIO()
    sf.write(buffer, augmented, sr, format="wav")

    example["audio"] = {
        "bytes": buffer.getvalue(),
        "path": None,
    }
    return example