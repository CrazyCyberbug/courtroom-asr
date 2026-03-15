import re
import inflect
import re
import pdfplumber
from num2words import num2words
import json

p = inflect.engine()
def extract_text(pdf_path):
  """
  pdf parser to extarct text from pdf
  """
  raw_text = []

  with pdfplumber.open(pdf_path) as pdf:
      for i, page in enumerate(pdf.pages):
        if i == 0:
          continue
        else:
          text = page.extract_text()
          if text:
              raw_text.append(text)

  raw_text = "\n".join(raw_text)
  return raw_text


  import re

def clean_text(text):
  """
  clean up non relavant part of transcript.
  Removes timestamps, transcription footer, transcription header, page numbers, [UNCLEAR] markers etc.
  """

  lines = text.split("\n")
  cleaned = []

  for line in lines:

      line = line.strip()

      # remove empty lines
      if not line:
          continue

      # remove timestamps
      if re.search(r"\d{1,2}:\d{2}\s*(AM|PM)\s*IST", line):
          continue

      # remove transcription footer
      if "Transcribed by" in line:
          continue

      # remove page numbers
      if re.fullmatch(r"\d+", line):
          continue

      # remove leading numbering
      line = re.sub(r"^\d+\s*", "", line)

      # remove speaker labels
      # line = re.sub(r"^[A-Z .']+:\s*", "", line)

      # remove unclear markers
      line = line.replace("[UNCLEAR]", "")

      line = line.replace("END OF DAY’S PROCEEDINGS", "")

      cleaned.append(line)

  return cleaned

def extract_speaker_level_dialog(lines):
  speaker_pattern = re.compile(r'^([A-Z .]+):')
  records = []
  current_speaker = None
  buffer = []

  for line in lines:

      line = line.strip()

      # remove line numbers like "12"
      if re.match(r'^\d+$', line):
          continue

      match = speaker_pattern.match(line)

      if match:
          # save previous speaker block
          if current_speaker:
              records.append({
                  "speaker": current_speaker,
                  "text": " ".join(buffer).strip()
              })

          current_speaker = match.group(1).strip()
          buffer = [line.split(":",1)[1].strip()]

      else:
          buffer.append(line)

  # save last
  if current_speaker:
      records.append({
          "speaker": current_speaker,
          "text": " ".join(buffer).strip()
      })
  return records

def normalize_numbers(text):
  def replace(match):
      num = match.group()
      try:
          return num2words(int(num))
      except:
          return num

  return re.sub(r'\d+', replace, text)

def normalize_legal_numbers(text):

    text = re.sub(r'(\d+)\((\d+)\)', lambda m: f"{num2words(int(m.group(1)))} {num2words(int(m.group(2)))}", text)

    return text

def normalize_text(text):

    text = text.lower()

    # convert numbers to words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)

    # remove section-style numbers like 10(2)
    text = re.sub(r'\(\d+\)', '', text)

    # remove hyphens
    text = text.replace('-', ' ')

    # remove ellipsis
    text = text.replace('...', ' ')

    # remove punctuation except apostrophe
    text = re.sub(r"[^a-z' ]+", ' ', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def split_records_to_sentences(records):
    """
    Split speaker level records into sentence level records.
    """

    sentence_records = []

    sentence_splitter = re.compile(r'(?<=[.!?])\s+')

    for record in records:

        speaker = record["speaker"]
        text = record["text"]

        sentences = sentence_splitter.split(text)

        for sent in sentences:

            sent = sent.strip()

            if not sent:
                continue

            # remove trailing punctuation
            # sent = re.sub(r"[.!?]+$", "", sent)

            sentence_records.append({
                "speaker": speaker,
                "text": sent
            })

    return sentence_records

def split_into_meaningful_segments(records, min_words=8, max_words=40):

    processed_records = []

    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    BAD_ENDINGS = {
    "and", "but", "so", "because", "however",
    "therefore", "thus", "then", "also", "yet"
    }


    for record in records:

      speaker = record["speaker"]
      text = record["text"].strip()

      sentences = sentence_splitter.split(text)

      buffer = []
      word_count = 0

      for sent in sentences:

          sent = sent.strip()
          if not sent:
              continue

          words = sent.split()
          buffer.append(sent)
          word_count += len(words)

          last_word = words[-1].lower().strip(".,!?;:")

          # check if segment should continue
          continue_segment = (
              word_count < min_words or
              last_word in BAD_ENDINGS
          )

          # finalize segment if conditions satisfied
          if not continue_segment or word_count >= max_words:

              processed_records.append({
                  "speaker": speaker,
                  "text": " ".join(buffer)
              })

              buffer = []
              word_count = 0

    # flush remaining buffer
    if buffer:
        processed_records.append({
            "speaker": speaker,
            "text": " ".join(buffer)
        })

    return processed_records

# main pipeline
def preprocess_transcripts(pdf_path):
  text = extract_text(pdf_path)
  lines = clean_text(text)
  records = extract_speaker_level_dialog(lines)
  text = "".join([record["text"]for record in records])
  text = normalize_numbers(text)
  text = normalize_legal_numbers(text)
  text = re.sub(r"[^\w\s']", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  text = normalize_text(text)
  records = split_records_to_sentences()
  clean_records = split_into_meaningful_segments(records)
  return text, records, clean_records