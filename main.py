"""
main.py: Multi-label emotion classifier for [anger, fear, joy, sadness, surprise]

Provides two modes:
  - train: fine-tunes a RoBERTa model on `track-a.csv`, saves best checkpoint & thresholds
  - predict: loads saved model/thresholds, predicts on input CSV, writes out emotion columns

Usage:
  python main.py train
  python main.py predict --input_csv PATH --output_csv OUTPUT
"""

import os
# Restrict PyTorch to GPU index 0 if available (set via CUDA_VISIBLE_DEVICES)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable CodeCarbon telemetry logs to suppress emission tracking output
os.environ["CODECARBON_OFF"] = "1"

import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.nn.functional import sigmoid
from torch.nn import BCEWithLogitsLoss
from datasets import Dataset as HFDataset

# ----------------------------------------------------------------------------------------------------------------------
# Constants & Globals
# ----------------------------------------------------------------------------------------------------------------------

# Column names for multi-label output
LABEL_COLUMNS = ["anger", "fear", "joy", "sadness", "surprise"]

# Detect and set device: GPU if available, else fallback to MPS or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: mps")
else:
    device = torch.device("cpu")
    print("Using device: cpu")

# Initialize shared tokenizer & base model; model head is reinitialized for 5-label classification
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
).to(device)

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------

def tokenize(batch):
    """
    Tokenize a batch of text for the model.

    Args:
        batch (dict): batch with key 'text', a list of strings.
    Returns:
        dict: tokenized inputs with padding/truncation to length 128.
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


def prepare_dataset(df: pd.DataFrame, label_cols: list) -> HFDataset:
    """
    Convert a pandas DataFrame into a Hugging Face Dataset with tokenized inputs and label tensors.

    Args:
        df (pd.DataFrame): must contain 'text' column and one column per label in label_cols.
        label_cols (list): list of label column names.

    Returns:
        HFDataset: formatted for Trainer (torch tensors).
    """
    df = df.copy()
    ds = HFDataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)

    # Build float32 labels for multi-label BCE
    labels = np.array(df[label_cols].values, dtype=np.float32)
    ds = ds.add_column("labels", labels.tolist())

    # Only keep the needed columns for training/inference
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return ds


class WeightedTrainer(Trainer):
    """
    Custom Trainer that overrides compute_loss to apply per-label positive weights.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels and move to device
        labels = inputs.pop("labels").to(device)
        # Move input_ids and attention_mask to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        # BCE loss with positive-class weighting
        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred) -> dict:
    """
    Computes micro-averaged F1 score for multi-label predictions.

    Args:
        pred: Prediction object from Trainer.
    Returns:
        dict: {'f1': float}
    """
    labels = pred.label_ids
    logits = pred.predictions
    probs = sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()

    f1 = f1_score(labels, preds, average="micro")
    return {"f1": f1}

# ----------------------------------------------------------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    Fine-tune the RoBERTa model on `track-a.csv`, calibrate thresholds, and save artifacts.
    """
    # Load & clean data
    df = pd.read_csv("track-a.csv")
    df["text"] = df["text"].fillna("")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Compute positive-class weights: num_neg / num_pos per label
    pos_counts = train_df[LABEL_COLUMNS].sum().values
    neg_counts = len(train_df) - pos_counts
    global pos_weight
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float)

    # Build HF datasets
    hf_train = prepare_dataset(train_df, LABEL_COLUMNS)
    hf_val   = prepare_dataset(val_df, LABEL_COLUMNS)

    # TrainingArguments config
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42,
        report_to="none",
        logging_steps=50,
        logging_dir="logs"
    )

    # Initialize our custom weighted trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train & save best checkpoint
    trainer.train()

    # Post-training: calibrate optimal thresholds per label on validation set
    val_outputs = trainer.predict(hf_val)
    val_logits  = val_outputs.predictions
    val_probs   = sigmoid(torch.tensor(val_logits)).numpy()
    val_labels  = val_outputs.label_ids

    best_thresholds = []
    for i in range(len(LABEL_COLUMNS)):
        p, r, thresh = precision_recall_curve(val_labels[:, i], val_probs[:, i])
        f1_scores = 2 * p * r / (p + r + 1e-8)
        best_thresholds.append(float(thresh[f1_scores.argmax()]))
    print("Calibrated thresholds:", best_thresholds)

    # Persist thresholds and model
    torch.save(best_thresholds, "thresholds.pt")
    trainer.save_model("best_model")

# ----------------------------------------------------------------------------------------------------------------------
# Inference routine
# ----------------------------------------------------------------------------------------------------------------------
def predict(csv_path: str, thresholds_path: str = "thresholds.pt") -> np.ndarray:
    """
    Load saved model & thresholds, run multi-label prediction on new CSV, return binary array.

    Args:
        csv_path (str): path to input CSV file with 'text' column
        thresholds_path (str): path to saved thresholds .pt file
    Returns:
        np.ndarray: shape (n_samples, 5) of {0,1} predictions
    """
    # Load thresholds into numpy array
    thresholds = torch.load(thresholds_path)
    thresholds = np.array(thresholds, dtype=np.float32)

    # Reload fine-tuned model
    tokenizer_ = RobertaTokenizerFast.from_pretrained("roberta-large")
    model_     = RobertaForSequenceClassification.from_pretrained("best_model").to(device)

    # Prepare input dataset
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("")
    ds = HFDataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Use a plain Trainer for predictions
    trainer = Trainer(model=model_)
    outputs = trainer.predict(ds)
    probs   = sigmoid(torch.tensor(outputs.predictions)).numpy()

    # Apply calibrated thresholds
    return (probs >= thresholds).astype(int)

# ----------------------------------------------------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict multi-label emotion classifier")
    parser.add_argument("mode", choices=["train", "predict"], help="Operation mode")
    parser.add_argument("--input_csv", type=str, help="Input CSV for prediction")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.mode == "train":
        main()
    else:
        assert args.input_csv, "--input_csv is required in predict mode"
        preds = predict(args.input_csv)
        df_out = pd.read_csv(args.input_csv)
        for idx, col in enumerate(LABEL_COLUMNS):
            df_out[col] = preds[:, idx]
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
