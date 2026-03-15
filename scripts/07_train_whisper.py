from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk
import torch
import evaluate
import os

CHECKPOINT_DIR = "/content/drive/MyDrive/asr_checkpoints_latest"
train_path = "/content/drive/MyDrive/whisper_data/train_features"
test_path  = "/content/drive/MyDrive/whisper_data/test_features"


DRIVE_PATH = "/content/drive"
if not os.path.exists(DRIVE_PATH):
    from google.colab import drive
    drive.mount(DRIVE_PATH)


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print("Checkpoint directory:", CHECKPOINT_DIR)



ds_train = load_from_disk(train_path)
ds_test  = load_from_disk(test_path)

# ── setup model ───────────────────────────────────────────────────────

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.language = "en"
model.generation_config.task = "transcribe"

# Keep only the columns the trainer needs — drop audio/text/duration to prevent
# the dataloader from pulling large raw columns into RAM each batch
KEEP_COLS = {"input_features", "labels"}
ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in KEEP_COLS])
ds_test  = ds_test.remove_columns([c for c in ds_test.column_names  if c not in KEEP_COLS])

# Tell HuggingFace datasets to memory-map rather than cache in RAM
ds_train = ds_train.with_format("numpy")   # keeps Arrow mmap; avoids Python-list copies
ds_test  = ds_test.with_format("numpy")

print(f"Train: {len(ds_train):,} examples | Eval: {len(ds_test):,} examples")

# ── Collator ──────────────────────────────────────────────────────────────────
class WhisperCollator:
    def __call__(self, batch):
        input_features = torch.stack([
            torch.tensor(x["input_features"], dtype=torch.float32)
            for x in batch
        ])
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in batch],
            batch_first=True,
            padding_value=-100
        )
        return {"input_features": input_features, "labels": labels}

data_collator = WhisperCollator()

# ── WER metric ────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    # Replace padding token so the decoder doesn't choke on -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}

# ── Training arguments ────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,

    # Evaluation
    eval_strategy="steps",       
    eval_steps=500,
    predict_with_generate=True,

    # Best model tracking
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    # Batch / gradient
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,

    # Optimiser
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,

    # Checkpointing
    save_steps=500,
    save_total_limit=3,

    # Dataloader — pin_memory=False prevents a second GPU-side copy of each batch;
    # prefetch_factor=2 keeps the pipeline fed without buffering many batches at once
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    dataloader_prefetch_factor=2,

    logging_steps=50,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ── Resume from latest checkpoint if one exists ───────────────────────────────
checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if "checkpoint" in f]
    if checkpoints:
        checkpoint = os.path.join(CHECKPOINT_DIR, sorted(checkpoints)[-1])
        print("Resuming from:", checkpoint)
        
        
if __name__ == "__main__":

    trainer.train(resume_from_checkpoint=checkpoint)