import os
import torch
import evaluate
from transformers import Seq2SeqTrainingArguments
from datasets import load_from_disk, save_to_disk, concatenate_datasets
from utils.audio_augmentation import augment_audio
from utils.text import normalize_text

wer_metric = evaluate.load("wer")

def train_test_split(ds, test_size =0.1):
    ds_split = ds.train_test_split(
    test_size = test_size,
    seed=42)
    
    ds_train_split = ds_split["train"]
    ds_test_split = ds_split["test"]
    return ds_train_split, ds_test_split

def prepare_features(example, processor):

    audio = example["audio"]

    example["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=16000
    ).input_features[0]

    example["labels"] = processor.tokenizer(
        example["text"]
    ).input_ids

    return example

def compute_metrics(pred, processor):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}

def add_data_augmentation(ds, save_first = True):
    
    # map to data points
    augmented_set = ds.map(
    augment_audio,
    num_proc=1,
    writer_batch_size=50,       # small batches → bounded RAM
    keep_in_memory=True,       # stream to disk instead of RAM
    load_from_cache_file=False,
    )
    
    if save_first:
        augmented_set.save_to_disk("augmented_set")
        augmented_set = load_from_disk("augmented_set")
    
    # 2x orginal data
    ds_train = concatenate_datasets([ds, augmented_set])
    return ds_train

def ds_normalize_map(ds):
    def normalize_map(example):
        example["text"] = normalize_text(example["text"])
        return example
    
    ds = ds.map(normalize_map, num_proc=2)
    return ds
    
    
    