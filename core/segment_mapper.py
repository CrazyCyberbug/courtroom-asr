import re
import json
import inflect
import torchaudio
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
from IPython.display import Audio, display
from datasets import Dataset, Audio
from huggingface_hub import login



from config import BASE_DIR

p = inflect.engine()

def normalize_word(w: str) -> str:
    w = w.lower().strip()
    w = re.sub(r"[^a-z']+", "", w)
    return w

def normalize_record_text(text: str) -> List[str]:
    """Tokenize + normalize a transcript sentence into a word list."""
    text = re.sub(r'\d+', lambda m: p.number_to_words(m.group()), text)
    text = text.lower().replace('-', ' ')
    text = re.sub(r"[^a-z' ]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return [w for w in text.split() if w]

def normalize_wts_word(w: str) -> str:
    return normalize_word(w)

def build_record_token_sequence(
    records: List[Dict],
) -> Tuple[List[str], List[int]]:
    """
    Returns
    -------
    tokens       : flat list of normalized words across all records
    token_to_rec : maps each token index → index in `records`
    """
    tokens, token_to_rec = [], []
    for rec_idx, rec in enumerate(records):
        for w in normalize_record_text(rec["text"]):
            tokens.append(w)
            token_to_rec.append(rec_idx)
    return tokens, token_to_rec

def word_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

def _find_seed_position(
    seed_words: List[str],
    rec_tokens: List[str],
    *,
    start_hint: int = 0,        # never look before this position (monotonicity)
    search_window: int = None,  # None → search to end of transcript
) -> int:
    """
    Slide `seed_words` over `rec_tokens[start_hint:]` to find the best
    starting offset for this chunk.

    Parameters
    ----------
    start_hint    : token index in the FULL token list where the previous chunk
                    finished — we never search before this point.
    search_window : if set, cap the search at start_hint + search_window tokens.
                    Leave as None to always scan to the end of the transcript.
    """
    if not seed_words:
        return start_hint

    n = len(rec_tokens)
    k = len(seed_words)

    if search_window is not None:
        search_end = min(start_hint + search_window, n - k + 1)
    else:
        search_end = n - k + 1

    search_end = max(search_end, start_hint + 1)   # at least one position

    best_pos, best_score = start_hint, -1.0

    for start in range(start_hint, search_end):
        score = sum(
            word_sim(seed_words[i], rec_tokens[start + i])
            for i in range(min(k, n - start))
        )
        if score > best_score:
            best_score = score
            best_pos = start

    return best_pos

def align_wts_to_records(
    word_timestamps: List[Dict],
    records: List[Dict],
    *,
    sim_threshold: float = 0.75,
    greedy_window: int = 40,               # per-word lookahead in transcript tokens
    boundary_trim_start_sec: float = 150.0,
    boundary_trim_end_sec: float = 120.0,
    start_hint: int = 0,                   # (FIX) where previous chunk ended
) -> Tuple[List[Dict], int]:
    """
    Align word-level timestamps from ONE chunk to the sentence records.

    Returns
    -------
    results       : list of dicts with start_time, end_time, speaker, text, …
    last_rec_token: the last matched token index in the global token list
                    — pass this as `start_hint` for the next chunk.

    Important
    ---------
    • word_timestamps times are chunk-relative (start from ~0).
    • The caller is responsible for adding the chunk_offset to make them
      global before storing / displaying.
    """
    rec_tokens, token_to_rec = build_record_token_sequence(records)
    n_rec_tokens = len(rec_tokens)

    wts_norm = [normalize_wts_word(w["text"]) for w in word_timestamps]
    n_wts = len(wts_norm)

    if n_wts == 0 or n_rec_tokens == 0:
        return [], start_hint

    total_audio_dur = word_timestamps[-1]["end"]

    # ── Phase 1: find where this chunk starts inside the full transcript ──────
    # Use the first ~10 non-boundary WTS words as an anchor seed.
    seed_words = []
    for i in range(min(30, n_wts)):
        if word_timestamps[i]["start"] >= boundary_trim_start_sec or i < 5:
            seed_words.append(wts_norm[i])
        if len(seed_words) == 10:
            break

    # FIX: search_window=None → scans from start_hint to end of transcript
    best_start_pos = _find_seed_position(
        seed_words,
        rec_tokens,
        start_hint=start_hint,   # (FIX) never look before previous chunk's end
        search_window=None,      # (FIX) no artificial cap
    )

    # ── Phase 2: greedy sequential matching from best_start_pos ──────────────
    rec_ptr = best_start_pos
    token_matches: List[Optional[int]] = [None] * n_wts

    for wts_idx, wts_word in enumerate(wts_norm):
        if not wts_word:
            continue

        wts_time = word_timestamps[wts_idx]["start"]
        in_boundary = (
            wts_time < boundary_trim_start_sec
            or (total_audio_dur - word_timestamps[wts_idx]["end"]) < boundary_trim_end_sec
        )

        # Raise similarity bar for boundary words to reduce drift
        threshold = sim_threshold + 0.1 if in_boundary else sim_threshold

        window_end = min(rec_ptr + greedy_window, n_rec_tokens)
        best_pos, best_score = None, threshold

        for pos in range(rec_ptr, window_end):
            s = word_sim(wts_word, rec_tokens[pos])
            if s > best_score:
                best_score = s
                best_pos = pos

        if best_pos is not None:
            token_matches[wts_idx] = best_pos
            rec_ptr = best_pos + 1

    last_matched_token = rec_ptr   # carry forward for next chunk

    # ── Phase 3: aggregate → sentence timestamps ──────────────────────────────
    rec_wts_hits: Dict[int, List[Dict]] = {i: [] for i in range(len(records))}

    for wts_idx, tok_pos in enumerate(token_matches):
        if tok_pos is None:
            continue
        rec_idx = token_to_rec[tok_pos]
        rec_wts_hits[rec_idx].append(word_timestamps[wts_idx])

    results = []
    for rec_idx, hits in rec_wts_hits.items():
        if not hits:
            continue
        results.append({
            "speaker":       records[rec_idx]["speaker"],
            "text":          records[rec_idx]["text"],
            "start_time":    hits[0]["start"],
            "end_time":      hits[-1]["end"],
            "matched_words": len(hits),
            "confidence":    len(hits) / max(1, len(normalize_record_text(records[rec_idx]["text"]))),
            "rec_idx":       rec_idx,          # useful for debugging
        })

    results.sort(key=lambda r: r["start_time"])
    return results, last_matched_token

def smooth_boundaries(
    aligned: List[Dict],
    gap_fill_threshold: float = 0.5,
) -> List[Dict]:
    for i in range(len(aligned) - 1):
        gap = aligned[i + 1]["start_time"] - aligned[i]["end_time"]
        if 0 < gap < gap_fill_threshold:
            mid = (aligned[i]["end_time"] + aligned[i + 1]["start_time"]) / 2
            aligned[i]["end_time"] = mid
            aligned[i + 1]["start_time"] = mid
    return aligned

def align_chunk(
    word_timestamps: List[Dict],
    records: List[Dict],
    *,
    chunk_offset: float = 0.0,
    start_hint: int = 0,
    sim_threshold: float = 0.75,
    greedy_window: int = 250,
    boundary_trim_start_sec: float = 150.0,
    boundary_trim_end_sec: float = 120.0,
) -> Tuple[List[Dict], int]:
    """
    Align one chunk and optionally shift timestamps by chunk_offset.

    Returns (aligned_segments, last_matched_token_idx)
    Pass last_matched_token_idx as `start_hint` for the next chunk.

    NOTE: chunk_offset should be 0.0 when inspecting audio from the *chunk*
    file itself (times are chunk-relative). Set it to the true wall-clock
    offset only when building a global timeline.
    """
    aligned, last_token = align_wts_to_records(
        word_timestamps,
        records,
        sim_threshold=sim_threshold,
        greedy_window=greedy_window,
        boundary_trim_start_sec=boundary_trim_start_sec,
        boundary_trim_end_sec=boundary_trim_end_sec,
        start_hint=start_hint,
    )
    aligned = smooth_boundaries(aligned)

    for seg in aligned:
        seg["start_time"] += chunk_offset
        seg["end_time"]   += chunk_offset

    return aligned, last_token

def align_all_chunks(
    case_dir: Path,
    records: List[Dict],
    *,
    sim_threshold: float = 0.75,
    greedy_window: int = 40,
    boundary_trim_start_sec: float = 150.0,
    boundary_trim_end_sec: float = 120.0,
) -> Dict[int, List[Dict]]:
    """
    Iterate over all chunk_{i}_word_timestamps.json files found in
    case_dir/partial_chunks/, align each one, and return a dict:
        { chunk_id : [aligned_segments_with_global_times] }

    Chunks are processed in order; the global transcript pointer (start_hint)
    advances monotonically so later chunks never re-match earlier sentences.

    Missing word-timestamp files (failed alignment) are skipped gracefully.
    """
    partial_dir = case_dir / "partial_chunks"
    wts_files = sorted(
        partial_dir.glob("chunk_*_word_timestamps.json"),
        key=lambda p: int(re.search(r"chunk_(\d+)_word", p.name).group(1)),
    )

    all_results: Dict[int, List[Dict]] = {}
    global_token_ptr = 0       # monotonically advances across chunks
    global_time_offset = 0.0   # accumulated wall-clock offset

    for wts_path in wts_files:
        chunk_id = int(re.search(r"chunk_(\d+)_word", wts_path.name).group(1))
        print(f"Processing chunk {chunk_id} …", end=" ")

        with open(wts_path) as f:
            wts = json.load(f)

        if not wts:
            print("empty — skipped.")
            continue

        aligned, global_token_ptr = align_chunk(
            wts,
            records,
            chunk_offset=global_time_offset,
            start_hint=global_token_ptr,
            sim_threshold=sim_threshold,
            greedy_window=greedy_window,
            boundary_trim_start_sec=boundary_trim_start_sec,
            boundary_trim_end_sec=boundary_trim_end_sec,
        )

        chunk_duration = wts[-1]["end"]
        global_time_offset += chunk_duration

        all_results[chunk_id] = aligned
        print(f"{len(aligned)} sentences aligned.")

    return all_results

def inspect_chunk(
    case_dir: Path,
    records: List[Dict],
    chunk_id: int,
    *,
    n_segments: int = 10,
    skip_segments: int = 0,
    sim_threshold: float = 0.75,
    greedy_window: int = 40,
    boundary_trim_start_sec: float = 150.0,
    boundary_trim_end_sec: float = 120.0,
    start_hint: int = 0,
    silent: bool = True,
) -> Tuple[List[Dict], int]:
    """
    Align a single chunk and play audio slices for visual inspection
    inside a Jupyter / Colab notebook.

    Times are kept CHUNK-RELATIVE so the audio slice indices are correct
    when loading chunk_{chunk_id}.wav.

    Parameters
    ----------
    skip_segments : skip the first N sentences (handy to jump past boundary noise)
    start_hint    : pass the last token index from the previous chunk to
                    guarantee forward-only matching.

    Returns (aligned_segments, last_matched_token)
    """
    partial_dir = case_dir / "partial_chunks"
    wts_path    = partial_dir / f"chunk_{chunk_id}_word_timestamps.json"
    audio_path  = partial_dir / f"chunk_{chunk_id}.wav"

    if not wts_path.exists():
        print(f"[ERROR] word-timestamps file not found: {wts_path}")
        return [], start_hint

    if not audio_path.exists():
        print(f"[ERROR] audio file not found: {audio_path}")
        return [], start_hint

    with open(wts_path) as f:
        wts = json.load(f)

    # chunk_offset=0.0  →  keep chunk-relative times for audio slicing
    aligned, last_token = align_chunk(
        wts,
        records,
        chunk_offset=0.0,
        start_hint=start_hint,
        sim_threshold=sim_threshold,
        greedy_window=greedy_window,
        boundary_trim_start_sec=boundary_trim_start_sec,
        boundary_trim_end_sec=boundary_trim_end_sec,
    )

    print(f"\n{'═'*70}")
    print(f"  Chunk {chunk_id}  —  {len(aligned)} sentences aligned")
    print(f"{'═'*70}\n")

    if not aligned:
        print("No sentences aligned for this chunk.")
        return aligned, last_token


    # ── Print summary of all aligned sentences ────────────────────────────────
    if not silent:
      print("── Full alignment summary ──────────────────────────────────────────")
      for i, seg in enumerate(aligned):
          marker = "▶" if skip_segments <= i < skip_segments + n_segments else " "
          print(
              f" {marker} [{i:3d}] [{seg['start_time']:7.2f}s → {seg['end_time']:7.2f}s]"
              f"  conf={seg['confidence']:.2f}  {seg['speaker']}: {seg['text'][:55]}"
          )

      # ── Load audio once ───────────────────────────────────────────────────────
      waveform, sr = torchaudio.load(str(audio_path))
      chunk_duration_sec = waveform.shape[1] / sr
      print(f"\nAudio loaded — duration: {chunk_duration_sec:.1f}s  |  sample rate: {sr} Hz")

      # ── Play the requested window of segments ─────────────────────────────────
      window = aligned[skip_segments: skip_segments + n_segments]
      print(f"\n── Playing segments {skip_segments} → {skip_segments + len(window) - 1} ──\n")

      for i, seg in enumerate(window, start=skip_segments):
          start_sample = int(max(seg["start_time"], 0) * sr)
          end_sample   = int(min(seg["end_time"], chunk_duration_sec) * sr)

          if start_sample >= end_sample:
              print(f"  [Segment {i}] — invalid slice ({start_sample} ≥ {end_sample}), skipped.\n")
              continue

        # audio_slice = waveform[:, start_sample:end_sample]

        # print(f"  Segment {i}")
        # print(f"  [{seg['start_time']:.2f}s → {seg['end_time']:.2f}s]"
        #       f"  conf={seg['confidence']:.2f}  matched={seg['matched_words']} words")
        # print(f"  {seg['speaker']}: {seg['text']}\n")
        # display(Audio(audio_slice.numpy(), rate=sr))
        # print()

    return aligned, last_token

def inspect_all_chunks(
    case_dir: Path,
    records: List[Dict],
    *,
    n_segments: int = 5,
    **kwargs,
) -> Dict[int, List[Dict]]:
    """
    Convenience wrapper: inspect every chunk in order, forwarding the
    transcript pointer so each chunk starts where the previous one ended.
    """
    partial_dir = case_dir / "partial_chunks"
    wts_files = sorted(
        partial_dir.glob("chunk_*_word_timestamps.json"),
        key=lambda p: int(re.search(r"chunk_(\d+)_word", p.name).group(1)),
    )

    all_results = {}
    global_token_ptr = 0

    for wts_path in wts_files:
        chunk_id = int(re.search(r"chunk_(\d+)_word", wts_path.name).group(1))
        aligned, global_token_ptr = inspect_chunk(
            case_dir,
            records,
            chunk_id,
            n_segments=n_segments,
            start_hint=global_token_ptr,
            **kwargs,
        )
        all_results[chunk_id] = aligned

    return all_results

def stitch_to_cleaned_records(
    aligned: List[Dict],          # output of align with old records.json
    cleaned_records: List[Dict],  # the new merged records
    *,
    sim_threshold: float = 0.45,  # lower bar since we're matching multi-sentence spans
) -> List[Dict]:
    """
    Takes fine-grained aligned segments (from old records) and merges them
    to match the structure of cleaned_records.

    Strategy: for each cleaned record, greedily consume consecutive aligned
    segments whose combined text best matches the cleaned record text.
    """

    def text_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    stitched = []
    seg_ptr = 0
    n_segs = len(aligned)

    for cr in cleaned_records:
        if seg_ptr >= n_segs:
            break

        cr_text = cr["text"].strip()
        best_end_ptr = seg_ptr          # last seg index to include (inclusive)
        best_score = -1.0

        # Try consuming 1..N consecutive segments and pick the best match
        accumulated_text = ""
        for end_ptr in range(seg_ptr, min(seg_ptr + 8, n_segs)):
            accumulated_text += (" " if accumulated_text else "") + aligned[end_ptr]["text"].strip()
            score = text_sim(accumulated_text, cr_text)
            if score > best_score:
                best_score = score
                best_end_ptr = end_ptr

        if best_score < sim_threshold:
            # No good match found — this cleaned record was likely a dropped
            # short utterance; skip it without advancing seg_ptr
            continue

        # Merge the matched segment span
        span = aligned[seg_ptr: best_end_ptr + 1]
        stitched.append({
            "speaker":       cr["speaker"],
            "text":          cr["text"],
            "start_time":    span[0]["start_time"],
            "end_time":      span[-1]["end_time"],
            "matched_words": sum(s["matched_words"] for s in span),
            "confidence":    sum(s["matched_words"] for s in span) /
                             max(1, len(normalize_record_text(cr["text"]))),
        })
        seg_ptr = best_end_ptr + 1

    return stitched

def inspect_stitched(
    case_dir: Path,
    aligned_clean: List[Dict],
    chunk_id: int,
    *,
    n_segments: int = 10,
    skip_segments: int = 0,
) -> None:
    audio_path = case_dir / "partial_chunks" / f"chunk_{chunk_id}.wav"
    waveform, sr = torchaudio.load(str(audio_path))
    chunk_duration_sec = waveform.shape[1] / sr

    print(f"\n{'═'*70}")
    print(f"  Chunk {chunk_id}  —  {len(aligned_clean)} stitched sentences")
    print(f"{'═'*70}\n")

    print("── Full alignment summary ──────────────────────────────────────────")
    for i, seg in enumerate(aligned_clean):
        marker = "▶" if skip_segments <= i < skip_segments + n_segments else " "
        print(
            f" {marker} [{i:3d}] [{seg['start_time']:7.2f}s → {seg['end_time']:7.2f}s]"
            f"seg {i}  conf={seg['confidence']:.2f}  {seg['speaker']}: {seg['text'][:55]}"
        )

    print(f"\nAudio loaded — duration: {chunk_duration_sec:.1f}s  |  sample rate: {sr} Hz")

    window = aligned_clean[skip_segments: skip_segments + n_segments]
    print(f"\n── Playing segments {skip_segments} → {skip_segments + len(window) - 1} ──\n")

    for i, seg in enumerate(window, start=skip_segments):
        start_sample = int(max(seg["start_time"], 0) * sr)
        end_sample   = int(min(seg["end_time"], chunk_duration_sec) * sr)

        if start_sample >= end_sample:
            print(f"  [Segment {i}] — invalid slice ({start_sample} ≥ {end_sample}), skipped.\n")
            continue

        print(f"  Segment {i}")
        print(f"  [{seg['start_time']:.2f}s → {seg['end_time']:.2f}s]"
              f"  conf={seg['confidence']:.2f}  matched={seg['matched_words']} words")
        print(f"  {seg['speaker']}: {seg['text']}\n")
        display(Audio(waveform[:, start_sample:end_sample].numpy(), rate=sr))
        print()
        
def clean_up_casename(case_name):
    normalized_case_name =  " ".join(case_name.split("_")[1:])
    return normalized_case_name.lower()

def is_chunk_available(case_pathial_chunks_path, chunk_id):
    chunk_ws = case_pathial_chunks_path / f"chunk_{chunk_id}_word_timestamps.json"
    return chunk_ws.exists()

def save_audio(audio_slice, audio_filename, output_folder = Path("courtroom_asr_audio")):
    output_folder.mkdir(parents=True, exist_ok=True)
    audio_path = output_folder / audio_filename
    torchaudio.save(str(audio_path), audio_slice, sample_rate=16000)

def build_dataset():
    rows = []

    for case_dir in BASE_DIR.iterdir():
        CASE_NAME  = case_dir.stem

        case_dir = BASE_DIR / CASE_NAME
        cleaned_case_name = clean_up_casename(case_dir.stem)
        case_pathial_chunks_path = case_dir / "partial_chunks"
        manifest_path = case_pathial_chunks_path / "manifest.json"
        output_dir = Path("courtroom_asr_audio")
        print(f"Processing {case_dir.stem}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        num_chunks = len(manifest)
        print(f"{num_chunks} identified")

        # Atomoic Sentence level transcripts
        with open(case_dir / "processed_trancripts"/ "records.json") as f:
            old_records = json.load(f)

        # Cleaner segment level transcripts
        with open(case_dir / "processed_trancripts"/ "cleaned_records.json") as f:
            cleaned_records = json.load(f)

        for chunk_id in range(1, num_chunks+1):

            if not is_chunk_available(case_pathial_chunks_path, chunk_id):
                print(f"Chunk {chunk_id} not available")
                continue

            aligned_fine, _ = inspect_chunk(
                case_dir, old_records, chunk_id,
                greedy_window=40,
                start_hint=0,
            )

            audio_path = case_pathial_chunks_path / f"chunk_{chunk_id}.wav"
            waveform, sr = torchaudio.load(str(audio_path))
            chunk_duration_sec = waveform.shape[1] / sr

            # Step 2: stitch up to cleaned_records granularity
            aligned_clean = stitch_to_cleaned_records(aligned_fine, cleaned_records)
            for i, seg in enumerate(aligned_clean):
                segment = seg.copy()
                start_sample = int(max(seg["start_time"], 0) * sr)
                end_sample   = int(min(seg["end_time"], chunk_duration_sec) * sr)

            if start_sample >= end_sample:
                continue

            # extract the audio slice
            audio_slice = waveform[:, start_sample:end_sample]

            # save the audio
            audio_file_name =  f"{case_dir.stem}_{chunk_id}_{i}.wav"
            audio_file_path = output_dir / audio_file_name
            save_audio(audio_slice, audio_file_name, output_dir)

            # save metadata for df
            segment["audio_path"] = str(audio_file_path)
            segment["chunk_id"]  = chunk_id
            segment["case_name"] = cleaned_case_name
            segment["segment_id"] = i
            rows.append(segment)
            
        df = pd.DataFrame(rows)
        
        return df
      
def push_to_hub(df, keep_columns: List[str], username:str = "CrazyCyberBug2", dataset_uri:str= "courtroom-asr-dataset"):
    login()    
    df = df[keep_columns]

    dataset = Dataset.from_pandas(df, preserve_index=False)

    # important: disable decoding
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset.push_to_hub(f"{username}/{dataset_uri}")
    