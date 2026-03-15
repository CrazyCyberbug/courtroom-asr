import torch
import torchaudio
from torchaudio.io import StreamReader
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks



model = load_silero_vad()

def run_vad_filter(audio_path, output_path, sample_rate = 16000):

    # create streaming reader
    streamer = StreamReader(audio_path)
    speech_audio = []

    # decode audio in chunks
    streamer.add_basic_audio_stream(
        frames_per_chunk=sample_rate * 30,   # 30s chunks
        sample_rate=sample_rate,
        num_channels=1
    )

    for (chunk,) in streamer.stream():

        # Ensure the audio chunk is a 1D tensor (mono).
        # If it's multi-dimensional (e.g., [channels, samples]), take the mean across channels.
        # Then squeeze to remove any singleton dimensions.
        if chunk.ndim > 1:
            wav = torch.mean(chunk, dim=1) # Take the mean across channels
        else:
            wav = chunk

        wav = wav.squeeze() # Ensure it's 1D, removing any remaining singleton dimensions

        if wav.numel() < sample_rate:
            continue

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sample_rate,
            min_silence_duration_ms=700,
            min_speech_duration_ms=250,
            threshold=0.5
        )

        if speech_timestamps:
            speech = collect_chunks(speech_timestamps, wav)
            speech_audio.append(speech)

    if len(speech_audio) == 0:
        raise RuntimeError("No speech detected")

    vad_audio = torch.cat(speech_audio).unsqueeze(0)

    torchaudio.save(output_path, vad_audio, sample_rate)