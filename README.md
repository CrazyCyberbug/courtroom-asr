# Courtroom ASR Pipeline

This repository contains an end-to-end pipeline for preparing and fine-tuning an Automatic Speech Recognition (ASR) model on **Supreme Court hearing audio**.

The project processes long courtroom recordings and transcript PDFs to produce **aligned audio-text segments** suitable for ASR training. The final dataset and fine-tuned model are hosted on the Hugging Face Hub.



# Overview

Courtroom recordings often present several challenges:

- Long audio sessions (often multiple hours)
- Multiple speakers
- Non-speech sections such as silence or procedural pauses
- Transcript formatting specific to legal proceedings

To address this, the repository implements a **modular data processing pipeline** that:

1. Downloads and organizes raw data
2. Cleans audio using speech detection
3. Extracts and normalizes transcript text
4. Splits long recordings into manageable chunks
5. Aligns transcripts to audio at the word level
6. Produces segment-level timestamps
7. Fine-tunes an ASR model



# Pipeline Steps

## 1. Inventory Creation
Downloads and organizes the dataset locally.

Each case contains:

- An audio recording (`.mp3`)
- A transcript document (`.pdf`)

These are stored in a structured inventory for further processing.


## 2. Voice Activity Detection (VAD)

Courtroom recordings contain long pauses and non-speech segments.

A VAD process filters these sections and produces:

    vad_filtered.wav

## 3. Text Extraction

Transcript PDFs are parsed to extract:

- Raw transcript text
- Speaker-level segments

Additional normalization is applied to handle:

- Court-specific clauses
- Date formatting
- Number normalization
- Speaker markers


## 4. Coarse Audio Chunking

Court recordings can be extremely long and difficult to align directly.

A custom coarse alignment process splits audio into **manageable chunks (~1 hour)** and extracts corresponding transcript portions.

larger audio lead to OOM errors.

This uses whipser to transcribe the first and last 10 secs of the 1 hour chunks. These refernce texts are used to find the relevant section of transcript


## 5. Forced Alignment

A CTC-based aligner is used to produce **word-level timestamps** between audio and transcript text.

Output format:

word | start_time | end_time


## 6. Segment Timestamp Mapping

Word timestamps are aggregated to produce **segment-level timestamps** corresponding to the speaker segments extracted from transcripts.

## 7. Model Training

The processed dataset is used to fine-tune **Whisper Small**.

Training setup:

- Model: Whisper Small
- Learning rate: 1e-5
- Warmup steps: 500
- Dataset: Supreme Court hearings

Evaluation metric: Word Error Rate (WER)

Final result: WER: 15.7% was acchieved 




---

# Outputs

### Dataset
Aligned audio-text dataset hosted on Hugging Face.

### Model
Fine-tuned ASR model for courtroom speech recognition.

---

# Running the Pipeline

Each step can be executed sequentially and independently via the scripts directory.

* python scripts/01_inventory_creation.py
* python scripts/02_vad_filtering.py
* python scripts/03_text_extraction.py
* python scripts/04_coarse_chunking.py
* python scripts/05_alignment.py
* python scripts/06_map_word_to_segments.py
* python scripts/07_train_whisper.py









