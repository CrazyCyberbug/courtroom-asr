import os
import torch
import evaluate
from datasets import load_dataset, Audio
from datasets import load_from_disk, save_to_disk
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from core.trainer import (compute_metrics,
                          add_data_augmentation,
                          ds_normalize_map,
                          prepare_features,
                          train_test_split) 



# ── prepare data ───────────────────────────────────────────────────────

# modify these
CHECKPOINT_DIR = "/content/drive/MyDrive/asr_checkpoints_latest"
train_path = "/content/drive/MyDrive/whisper_data/train_features"
test_path  = "/content/drive/MyDrive/whisper_data/test_features"

ds = load_dataset("CrazyCyberBug2/courtroom-asr-improved-chunking", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# filtering dataset
ds = ds.filter(
    lambda x: x["duration"] >= 1.0 and x['duration']<=40.0 and 0.9 <= x["confidence"] <= 1.4,
    num_proc=4
)

ds = ds_normalize_map(ds)
ds_augmented = add_data_augmentation(ds)
ds_train_split, ds_test_split = train_test_split(ds_augmented)



ds_train = ds_train_split.map(
    prepare_features,
    num_proc=1
)


ds_test = ds_test_split.map(
    prepare_features,
    num_proc=1

)

ds_train.save_to_disk(train_path)
ds_test.save_to_disk(test_path)



# ── setup model ───────────────────────────────────────────────────────

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.language = "en"
model.generation_config.task = "transcribe"


KEEP_COLS = {"input_features", "labels"}
ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in KEEP_COLS])
ds_test  = ds_test.remove_columns([c for c in ds_test.column_names  if c not in KEEP_COLS])


ds_train = ds_train.with_format("numpy")  
ds_test  = ds_test.with_format("numpy")

print(f"Train: {len(ds_train):,} examples | Eval: {len(ds_test):,} examples")


# ── Defining collator ───────────────────────────────────────────────────────


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

def call_compute_metrics(pred):
    return compute_metrics(pred, processor)

# ── setup trainer ───────────────────────────────────────────────────────

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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Resume from latest checkpoint if one exists
checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if "checkpoint" in f]
    if checkpoints:
        checkpoint = os.path.join(CHECKPOINT_DIR, sorted(checkpoints)[-1])
        print("Resuming from:", checkpoint)
        
        
if __name__ == "__main__":
    
# ── training loop ───────────────────────────────────────────────────────

    trainer.train(resume_from_checkpoint=checkpoint)