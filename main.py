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

# Emotion labels
LABEL_COLUMNS = ["anger", "fear", "joy", "sadness", "surprise"]

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def prepare_dataset(df, label_cols):
    df = df.copy()
    ds = HFDataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)
    labels = np.array(df[label_cols].values, dtype=np.float32)
    ds = ds.add_column("labels", labels.tolist())
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return ds

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    probs = sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    f1 = f1_score(labels, preds, average="micro")
    return {"f1": f1}

def main():
    # Load and split data
    df = pd.read_csv("track-a.csv")
    df["text"] = df["text"].fillna("")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Compute pos_weight for weighted loss
    pos_counts = train_df[LABEL_COLUMNS].sum().values
    neg_counts = len(train_df) - pos_counts
    global pos_weight
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float)

    # Initialize tokenizer and model
    global tokenizer, model
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification"
    )

    # Prepare datasets
    hf_train = prepare_dataset(train_df, LABEL_COLUMNS)
    hf_val   = prepare_dataset(val_df, LABEL_COLUMNS)

    # Training arguments
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42,
        report_to="none",
        logging_steps=50,
        logging_dir="logs"
    )

    # Initialize trainer with weighted loss and early stopping
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Fine-tune the model
    trainer.train()

    # Calibrate thresholds on validation set
    val_outputs = trainer.predict(hf_val)
    val_logits  = val_outputs.predictions
    val_probs   = sigmoid(torch.tensor(val_logits)).numpy()
    val_labels  = val_outputs.label_ids

    best_thresholds = []
    for i in range(len(LABEL_COLUMNS)):
        p, r, thresh = precision_recall_curve(val_labels[:, i], val_probs[:, i])
        f1_scores = 2 * p * r / (p + r + 1e-8)
        best_thresh = thresh[f1_scores.argmax()]
        best_thresholds.append(best_thresh)
    print("Calibrated thresholds:", best_thresholds)

    # Save thresholds and the best model
    torch.save(best_thresholds, "thresholds.pt")
    trainer.save_model("best_model")


def predict(csv_path, thresholds_path="thresholds.pt"):
    # Load model, tokenizer, and thresholds
    thresholds = torch.load(thresholds_path)
    tokenizer  = RobertaTokenizerFast.from_pretrained("best_model")
    model      = RobertaForSequenceClassification.from_pretrained("best_model")

    # Load and preprocess input data
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("")
    ds = HFDataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Predict and apply calibrated thresholds
    outputs = Trainer(model=model).predict(ds)
    probs   = sigmoid(torch.tensor(outputs.predictions)).numpy()
    preds   = (probs >= thresholds).astype(int)
    return preds

if __name__ == "__main__":
    main()
