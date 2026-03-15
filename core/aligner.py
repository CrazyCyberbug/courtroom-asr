import json
import torch
import faster_whisper
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8


def produce_alignments(audio_path, transcript_path, word_timestamps_path):
    with open(transcript_path, "r") as f:
        text = f.read()

    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    audio_waveform = faster_whisper.decode_audio(audio_path)

    audio_waveform = (
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device)
    )

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    del alignment_model
    torch.cuda.empty_cache()

    # Get the raw tokens and text from preprocess_text, which include '<star>' tokens
    tokens_with_stars, text_with_stars = preprocess_text(
        text,
        romanize=True,
        language="eng",
    )

    # Filter out the '<star>' tokens from the tokens used for encoding in get_alignments,
    # as they are typically not in the alignment_tokenizer's vocabulary.
    tokens_for_alignment = [token for token in tokens_with_stars if token != "<star>"]

    # Filter out the '<star>' tokens from the text used for post-processing,
    # to maintain a consistent length with the aligned segments and spans.
    text_for_postprocessing = [word for word in text_with_stars if word != "<star>"]

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_for_alignment, # Use filtered tokens for alignment
        alignment_tokenizer,
    )

    spans = get_spans(tokens_for_alignment, segments, blank_token) # Use filtered tokens for spans

    word_timestamps = postprocess_results(text_for_postprocessing, spans, stride, scores)
   
    with open(word_timestamps_path, "w") as f:
        json.dump(word_timestamps, f, indent = 2)
    
    return word_timestamps, word_timestamps_path
