import os
import re
import fitz
import json
import os
import time
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import BASE_DIR, DATASET_URL

MAX_RETRIES = 3


def get_dataset_csv(output_path = BASE_DIR / "dataset.csv"):
    os.makedirs(BASE_DIR, exist_ok=True)
    if not output_path.exists():
        cmd = ["wget", "-O", str(output_path), DATASET_URL]
        subprocess.run(cmd, check=False)
    return output_path
    
def extract_drive_file_id(url):
    """
    Extract file id from google drive url
    Example:
    https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None

def convert_dropbox_link(url):
    """
    change dl=0 -> dl=1
    """
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url

def download_audio(url, output_path):
    """
    Download using wget
    """
    cmd = ["wget", "-O", str(output_path), url]
    subprocess.run(cmd, check=False)

def sanitize_filename(name):
    return re.sub(r"[^\w\-]+", "_", str(name))

def download_transcript(url, transcript_url, output_path):
  """This code downloads the transcript files:
  Attempts to use `wget` command (faster) to download, on failure it falls back to `gdown` command."""



  for attempt in range(1, MAX_RETRIES + 1):

      print(f"WGET attempt {attempt}")

      cmd = ["wget", "-q", url, "-O", str(output_path)]
      subprocess.run(cmd)

      if output_path.exists() and output_path.stat().st_size > 1000:
          return True

      print("wget failed. Retrying...\n")
      time.sleep(2)


  print("Switching to gdown fallback")

  cmd = [
      "gdown",
      "--fuzzy",
      transcript_url,
      "-O",
      str(output_path)
  ]

  subprocess.run(cmd)

  if output_path.exists() and output_path.stat().st_size > 1000:
      return True

  return False

def clean_field(value, default):
    if pd.isna(value):
        return default
    value = str(value).strip()
    if value == "":
        return default
    value = re.sub(r"[^\w\-]", "_", value)   # remove problematic chars
    return value

def verify_audio(audio_path):

    if not audio_path or not os.path.exists(audio_path):
        return False, "Audio file missing"

    try:

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-show_streams",
            "-of", "json",
            str(audio_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        info = json.loads(result.stdout)

        duration = float(info["format"]["duration"])
        codec = info["streams"][0]["codec_name"]

        if duration < 1:
            return False, "Audio too short"

        return True, f"OK | {duration:.1f}s | codec={codec}"

    except Exception as e:
        return False, f"Unreadable audio: {e}"

def verify_pdf(pdf_path):

    if not pdf_path or not os.path.exists(pdf_path):
        return False, "PDF missing"

    try:
        doc = fitz.open(pdf_path)

        n_pages = len(doc)

        if n_pages == 0:
            doc.close()
            return False, "PDF empty"

        text = doc[0].get_text()

        doc.close()

        if len(text.strip()) < 20:
            return False, "PDF text extraction failed"

        return True, f"OK | {n_pages} pages"

    except Exception as e:
        return False, f"Unreadable PDF: {e}"

def build_dataset(df):

    metadata = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        case_number = clean_field(row["Case Number"], f"case_{idx}")
        case_name = clean_field(row["Case Name"], f"unknown_case")
        case_dir = BASE_DIR / case_name
        case_raw_dir = case_dir / "raw"
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(case_raw_dir, exist_ok=True)


        audio_path = case_raw_dir / f"recording.mp3"
        transcript_path = case_raw_dir / f"transcript.pdf"


        # Audio download
        audio_url = row["mp3 format link"]

        if pd.notna(audio_url):

            audio_url = convert_dropbox_link(audio_url)

            if not audio_path.exists():
                download_audio(audio_url, audio_path)

        else:
            audio_path = None


        # Transcript download
        transcript_url = row["Transcript Link"]
        sr_no = row["Sr. No."]
        case_name = sanitize_filename(row["Case Name"])

        if pd.isna(transcript_url):
            print("Missing transcript link\n")
            continue

        print(transcript_url)

        try:

          file_id = extract_drive_file_id(transcript_url)

          output_path = transcript_path

          url = f"https://drive.google.com/uc?export=download&id={file_id}"

          success = download_transcript(url, transcript_url, output_path)

          if not success:
              print(f"DOWNLOAD FAILED: {transcript_url}\n")
              continue

          print(f"\ncase {sr_no} : {case_name} | successfully downloaded")

          flag, message = verify_pdf(str(output_path))
          print(flag, message)
          print("\n")
        except Exception as e:
          print("ERROR down:", e)


        # Maintain Metadata
        row_dict = row.to_dict()

        row_dict["audio_path"] = str(audio_path) if audio_path else None
        try:
          flag, message = verify_audio(str(audio_path))
          print("AUDIO CHECK:", message, audio_path, "\n")
        except Exception as e:
          print(e)


        row_dict["transcript_path"] = str(transcript_path) if transcript_path else None
        try:
          flag, message = verify_pdf(str(transcript_path))
          print("TRANSCRIPT CHECK", message, str(transcript_path), "\n")
        except Exception as e:
          print(e)

        metadata.append(row_dict)


    metadata_df = pd.DataFrame(metadata)

    metadata_df.to_csv(BASE_DIR / "metadata.csv", index=False)

    return metadata_df

