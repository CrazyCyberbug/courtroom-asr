from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── lazy imports (only needed at runtime) ──────────────────────────────────
try:
    import whisper  # openai-whisper
except ImportError:
    whisper = None  # type: ignore

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None  # type: ignore

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz import process as rprocess
except ImportError:
    rfuzz = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    
    

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


# Common number words → digits (extend as needed)
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}

# Common ASR homophones / substitution errors
_ASR_CORRECTIONS = {
    r"\btheir\b": "there", r"\bthey're\b": "there",
    r"\byour\b": "you're",   # keep both surface forms normalised away
    r"\bits\b": "it's",
    r"\bto\b": "to", r"\btoo\b": "to", r"\btwo\b": "2",
    r"\bfor\b": "for", r"\bfour\b": "4",
    r"\bno\b": "know",  # intentionally collapse – helps matching
    r"\bnew\b": "knew",
    r"\bwrite\b": "right", r"\bwright\b": "right",
    r"\bwon\b": "one",
    r"\bsun\b": "son",
}

def normalise(text: str, *, numbers: bool = True, asr: bool = True) -> str:
    """
    Aggressive normalisation designed to maximise overlap between ASR output
    and a reference transcript that may use different conventions.

    Steps
    ─────
    1. Unicode NFKD → ASCII transliteration
    2. Lowercase
    3. Expand contractions minimally (can't → cant, etc.)
    4. Remove punctuation (keep spaces)
    5. Collapse whitespace
    6. (optional) Map number words → digits
    7. (optional) Collapse common ASR homophones
    """
    # 1. Unicode → ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 2. Lowercase
    text = text.lower()

    # 3. Expand contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)

    # 4. Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Number words → digits
    if numbers:
        words = text.split()
        words = [_NUM_WORDS.get(w, w) for w in words]
        text = " ".join(words)

    # 7. ASR homophone collapse
    if asr:
        for pattern, replacement in _ASR_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text)

    return text

def char_ngrams(text: str, n: int = 3) -> set:
    """Character n-gram set (after normalisation)."""
    return {text[i : i + n] for i in range(len(text) - n + 1)}

def word_ngrams(text: str, n: int = 2) -> list:
    """Word n-gram list."""
    tokens = text.split()
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def word_ngram_precision(hypothesis: str, reference: str, n: int = 2) -> float:
    """
    Fraction of hypothesis n-grams that appear in reference
    (unigram BLEU-style, clipped).
    """
    hyp_ngrams = word_ngrams(hypothesis, n)
    if not hyp_ngrams:
        return 0.0
    ref_ngrams_bag: dict = {}
    for ng in word_ngrams(reference, n):
        ref_ngrams_bag[ng] = ref_ngrams_bag.get(ng, 0) + 1

    clipped = 0
    hyp_bag: dict = {}
    for ng in hyp_ngrams:
        hyp_bag[ng] = hyp_bag.get(ng, 0) + 1

    for ng, cnt in hyp_bag.items():
        clipped += min(cnt, ref_ngrams_bag.get(ng, 0))

    return clipped / sum(hyp_bag.values())

@dataclass
class MatchResult:
    char_start: int          # character offset in full transcript
    char_end: int
    word_start: int          # word index in full transcript
    word_end: int
    score: float
    method_scores: dict = field(default_factory=dict)

def _tfidf_similarity(query: str, candidates: List[str]) -> np.ndarray:
    """Return cosine similarity between query and each candidate string."""
    if not _SKLEARN:
        return np.zeros(len(candidates))
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        matrix = vec.fit_transform([query] + candidates)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return sims
    except Exception:
        return np.zeros(len(candidates))

def find_boundary(
    query_raw: str,
    full_transcript: str,
    *,
    window_words: int = 80,
    stride_words: int = 10,
    char_ngram_n: int = 3,
    word_ngram_n: int = 2,
    weights: Optional[dict] = None,
    top_k_tfidf: int = 30,
) -> MatchResult:
    """
    Locate *query_raw* (Whisper transcription of a boundary segment) inside
    *full_transcript* using a 4-layer ensemble.

    Parameters
    ──────────
    query_raw      : raw Whisper output for the 30-s boundary audio
    full_transcript: the reference transcript (full document)
    window_words   : sliding window width in words
    stride_words   : step size for the sliding window
    char_ngram_n   : n for character n-gram Jaccard
    word_ngram_n   : n for word n-gram precision
    weights        : {'fuzzy': w1, 'char_ngram': w2, 'word_ngram': w3, 'tfidf': w4}
    top_k_tfidf    : run expensive TF-IDF only on the top-K fuzzy candidates

    Returns a MatchResult with character + word offsets in full_transcript.
    """
    if weights is None:
        weights = {"fuzzy": 0.30, "char_ngram": 0.25, "word_ngram": 0.25, "tfidf": 0.20}

    q_norm = normalise(query_raw)
    ft_norm = normalise(full_transcript)

    ft_words = ft_norm.split()
    q_words  = q_norm.split()

    if not ft_words or not q_words:
        return MatchResult(0, 0, 0, 0, 0.0)

    # ── build word-aligned character offset map ──────────────────────────────
    # Maps word index → (char_start, char_end) in the *original* full_transcript
    word_char_spans = _word_char_spans(full_transcript)

    # ── enumerate windows ────────────────────────────────────────────────────
    windows: List[Tuple[int, int, str]] = []   # (word_start, word_end, text)
    for ws in range(0, max(1, len(ft_words) - window_words + 1), stride_words):
        we = min(ws + window_words, len(ft_words))
        windows.append((ws, we, " ".join(ft_words[ws:we])))

    if not windows:
        windows = [(0, len(ft_words), ft_norm)]

    window_texts = [w[2] for w in windows]

    # ── Layer 1: rapidfuzz partial_ratio ─────────────────────────────────────
    if rfuzz is not None:
        fuzzy_scores = np.array([
            rfuzz.partial_ratio(q_norm, wt) / 100.0 for wt in window_texts
        ])
    else:
        # fallback: character overlap ratio
        fuzzy_scores = np.array([
            len(set(q_norm) & set(wt)) / max(len(set(q_norm) | set(wt)), 1)
            for wt in window_texts
        ])

    # ── Layer 2: character n-gram Jaccard ─────────────────────────────────────
    q_cng = char_ngrams(q_norm, char_ngram_n)
    cng_scores = np.array([jaccard(q_cng, char_ngrams(wt, char_ngram_n))
                           for wt in window_texts])

    # ── Layer 3: word n-gram precision ────────────────────────────────────────
    wng_scores = np.array([word_ngram_precision(q_norm, wt, word_ngram_n)
                           for wt in window_texts])

    # ── Layer 4: TF-IDF cosine (only on top-K by fuzzy to save time) ─────────
    tfidf_scores = np.zeros(len(windows))
    if _SKLEARN and len(windows) > 0:
        top_k_idx = np.argsort(fuzzy_scores)[::-1][:top_k_tfidf]
        candidate_texts = [window_texts[i] for i in top_k_idx]
        sims = _tfidf_similarity(q_norm, candidate_texts)
        for rank, orig_idx in enumerate(top_k_idx):
            tfidf_scores[orig_idx] = sims[rank]

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble = (
        weights["fuzzy"]      * fuzzy_scores +
        weights["char_ngram"] * cng_scores   +
        weights["word_ngram"] * wng_scores   +
        weights["tfidf"]      * tfidf_scores
    )

    best_idx = int(np.argmax(ensemble))
    ws_best, we_best, _ = windows[best_idx]

    # Map word indices back to character spans in the *original* transcript
    cs = word_char_spans[ws_best][0] if ws_best < len(word_char_spans) else 0
    ce = word_char_spans[min(we_best - 1, len(word_char_spans) - 1)][1]

    return MatchResult(
        char_start=cs,
        char_end=ce,
        word_start=ws_best,
        word_end=we_best,
        score=float(ensemble[best_idx]),
        method_scores={
            "fuzzy":      float(fuzzy_scores[best_idx]),
            "char_ngram": float(cng_scores[best_idx]),
            "word_ngram": float(wng_scores[best_idx]),
            "tfidf":      float(tfidf_scores[best_idx]),
        },
    )

def _word_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char offsets for each whitespace-delimited token."""
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end()))
    return spans

def transcribe_segment(
    audio_array: np.ndarray,
    sr: int,
    model,
    *,
    start_sec: float = 0.0,
    duration_sec: float = 30.0,
    language: Optional[str] = None,
) -> str:
    
    start_s = int(start_sec * sr)
    end_s   = int((start_sec + duration_sec) * sr)
    segment = audio_array[start_s:end_s]

    # Whisper expects 16 kHz float32
    if sr != 16000:
        import librosa
        segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

    kwargs = {"language": language} if language else {}
    result = model.transcribe(segment, **kwargs)
    return result["text"].strip()

@dataclass
class AudioChunk:
    index: int
    start_sec: float
    end_sec: float
    audio: np.ndarray          # float32, 16 kHz
    transcript_slice: str      # corresponding portion of the reference transcript
    char_start: int            # offset in full transcript
    char_end: int
    boundary_start_score: float
    boundary_end_score: float

def chunk_audio_and_transcript(
    audio_path: str,
    full_transcript: str,
    *,
    chunk_duration_sec: float = 1800.0,   # 30 min
    probe_duration_sec: float = 30.0,     # Whisper probe length
    whisper_model_size: str = "base",
    language: Optional[str] = None,
    window_words: int = 80,
    stride_words: int = 10,
    overlap_words: int = 20,              # extend each slice by N words for safety
    match_weights: Optional[dict] = None,
    verbose: bool = True,
) -> List[AudioChunk]:
    """
    End-to-end chunker.

    1. Load audio (via pydub or torchaudio fallback).
    2. Split into chunks of *chunk_duration_sec*.
    3. Whisper-transcribe first/last *probe_duration_sec* of each chunk.
    4. Use multi-layer search to locate transcript boundaries.
    5. Return list of AudioChunk objects.
    """
    # ── load audio ────────────────────────────────────────────────────────────
    audio_np, sr = _load_audio(audio_path)
    total_sec = len(audio_np) / sr
    if verbose:
        print(f"[chunker] Loaded {audio_path!r}: {total_sec:.1f} s @ {sr} Hz")

    # ── load whisper ──────────────────────────────────────────────────────────
    assert whisper is not None, "openai-whisper not installed"
    if verbose:
        print(f"[chunker] Loading Whisper '{whisper_model_size}' …")
    wmodel = whisper.load_model(whisper_model_size)

    # ── build chunk boundaries ────────────────────────────────────────────────
    chunk_starts = np.arange(0, total_sec, chunk_duration_sec)
    chunks_out: List[AudioChunk] = []

    # We search each boundary in only the *relevant* half of the transcript to
    # keep search fast and avoid false matches on repeated phrases.
    n_chunks = len(chunk_starts)
    transcript_words = normalise(full_transcript).split()
    words_per_chunk  = max(1, len(transcript_words) // max(n_chunks, 1))

    for idx, t_start in enumerate(chunk_starts):
        t_end   = min(t_start + chunk_duration_sec, total_sec)
        s_start = int(t_start * sr)
        s_end   = int(t_end   * sr)
        chunk_audio = audio_np[s_start:s_end]

        if verbose:
            print(f"\n[chunk {idx}] {t_start:.0f}–{t_end:.0f} s")

        # ── probe first 30 s ─────────────────────────────────────────────────
        first_text = transcribe_segment(
            chunk_audio, sr, wmodel,
            start_sec=0.0, duration_sec=probe_duration_sec, language=language,
        )
        if verbose:
            print(f"  ↳ first-probe: {first_text[:80]!r}")

        # ── probe last 30 s ───────────────────────────────────────────────────
        last_start = max(0.0, (t_end - t_start) - probe_duration_sec)
        last_text = transcribe_segment(
            chunk_audio, sr, wmodel,
            start_sec=last_start, duration_sec=probe_duration_sec, language=language,
        )
        if verbose:
            print(f"  ↳ last-probe:  {last_text[:80]!r}")

        # ── restrict search region in transcript ──────────────────────────────
        # Allow ±50 % of a chunk's expected word-span around the estimated
        # position to avoid false matches on long repetitive transcripts.
        search_margin = int(words_per_chunk * 0.5)
        region_start_w = max(0, idx * words_per_chunk - search_margin)
        region_end_w   = min(len(transcript_words),
                             (idx + 1) * words_per_chunk + search_margin)

        region_words   = transcript_words[region_start_w:region_end_w]
        region_text    = " ".join(region_words)

        # Reconstruct char offset of region_start_w in full_transcript
        ft_word_spans  = _word_char_spans(full_transcript)

        def region_to_full_char(region_word_idx: int) -> int:
            abs_word = region_start_w + region_word_idx
            if abs_word >= len(ft_word_spans):
                return len(full_transcript)
            return ft_word_spans[abs_word][0]

        # ── match start boundary ─────────────────────────────────────────────
        m_start = find_boundary(
            first_text, region_text,
            window_words=window_words,
            stride_words=stride_words,
            weights=match_weights,
        )

        # ── match end boundary ───────────────────────────────────────────────
        m_end = find_boundary(
            last_text, region_text,
            window_words=window_words,
            stride_words=stride_words,
            weights=match_weights,
        )

        # ── extract transcript slice (with word overlap for safety) ───────────
        slice_word_start = max(0, m_start.word_start - overlap_words)
        slice_word_end   = min(len(region_words), m_end.word_end + overlap_words)

        # Recover from original (un-normalised) full_transcript
        abs_word_start = region_start_w + slice_word_start
        abs_word_end   = region_start_w + slice_word_end

        char_s = ft_word_spans[abs_word_start][0] if abs_word_start < len(ft_word_spans) else 0
        char_e = ft_word_spans[min(abs_word_end, len(ft_word_spans)) - 1][1]

        transcript_slice = full_transcript[char_s:char_e]

        if verbose:
            print(f"  ↳ transcript slice: chars [{char_s}:{char_e}]  "
                  f"score_start={m_start.score:.3f}  score_end={m_end.score:.3f}")

        chunks_out.append(AudioChunk(
            index=idx,
            start_sec=float(t_start),
            end_sec=float(t_end),
            audio=chunk_audio,
            transcript_slice=transcript_slice,
            char_start=char_s,
            char_end=char_e,
            boundary_start_score=m_start.score,
            boundary_end_score=m_end.score,
        ))

    return chunks_out

def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono numpy array."""
    if AudioSegment is not None:
        seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(seg.array_type).max
        return samples, 16000

    # torchaudio fallback
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze().numpy().astype(np.float32), 16000
    except Exception as e:
        raise RuntimeError(
            "Install pydub (+ ffmpeg) or torchaudio to load audio."
        ) from e

def locate_text_in_transcript(
    query: str,
    full_transcript: str,
    **kwargs,
) -> MatchResult:
    """
    Convenience wrapper: find where *query* sits in *full_transcript*.
    Useful for unit-testing the matching logic without loading audio.
    """
    return find_boundary(query, full_transcript, **kwargs)

def save_chunks(
    audio_path: str,
    transcript_path: str,
    out_dir: str,
    chunk_minutes: float = 30.0,
    probe_sec: float = 30.0,
    whisper_model: str = "base",
    language: str | None = None,
    overlap_words: int = 20,
    verbose: bool = True,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_transcript = Path(transcript_path).read_text(encoding="utf-8")

    chunks = chunk_audio_and_transcript(
        audio_path,
        full_transcript,
        chunk_duration_sec=chunk_minutes * 60,
        probe_duration_sec=probe_sec,
        whisper_model_size=whisper_model,
        language=language,
        overlap_words=overlap_words,
        verbose=verbose,
    )

    manifest = []

    for c in chunks:
        label = f"chunk_{c.index + 1}"   # 1-indexed

        # ── write wav ─────────────────────────────────────────────────────────
        wav_path = out / f"{label}.wav"
        sf.write(str(wav_path), c.audio, 16000, subtype="PCM_16")

        # ── write transcript slice ────────────────────────────────────────────
        txt_path = out / f"{label}_transcript.txt"
        txt_path.write_text(c.transcript_slice, encoding="utf-8")

        # ── manifest entry ────────────────────────────────────────────────────
        entry = {
            "chunk": label,
            "start_sec": round(c.start_sec, 3),
            "end_sec":   round(c.end_sec,   3),
            "char_start": c.char_start,
            "char_end":   c.char_end,
            "boundary_score_start": round(c.boundary_start_score, 4),
            "boundary_score_end":   round(c.boundary_end_score,   4),
            "wav":        str(wav_path),
            "transcript": str(txt_path),
        }
        manifest.append(entry)

        if verbose:
            dur = c.end_sec - c.start_sec
            words = len(c.transcript_slice.split())
            print(
                f"  ✓ {label}  {dur/60:.1f} min  |  "
                f"{words} transcript words  |  "
                f"match scores ({c.boundary_start_score:.3f}, {c.boundary_end_score:.3f})"
            )

    # ── write manifest ────────────────────────────────────────────────────────
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[done] {len(chunks)} chunk(s) → {out}/")
    print(f"       manifest saved to {manifest_path}")

    return manifest
