import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.nn.functional import sigmoid
from datasets import Dataset as HFDataset

# Emotion labels
LABEL_COLUMNS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# === Training Phase ===
# Load and preprocess training data
train_df = pd.read_csv('track-a.csv')
train_df['text'] = train_df['text'].fillna('')

# Split data into train and validation sets (80/20)
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42
)

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
)

# Tokenization helper
def tokenize(batch):
    return tokenizer(
        batch['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=128
    )

# Prepare Hugging Face datasets - FIXED
def prepare_dataset(df):
    # Create Hugging Face Dataset
    ds = HFDataset.from_pandas(df)
    
    # Tokenize text
    ds = ds.map(tokenize, batched=True)
    
    # Create labels tensor - CRITICAL FIX
    labels = np.array(df[LABEL_COLUMNS], dtype=np.float32)
    ds = ds.add_column('labels', labels.tolist())
    
    # Set format for training
    ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']  # Fixed columns
    )
    return ds

hf_train = prepare_dataset(train_df)
hf_val = prepare_dataset(val_df)

# Metrics calculation function
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    # Apply sigmoid and threshold at 0.5
    probs = sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    
    # Calculate micro F1 score
    f1 = f1_score(labels, preds, average='micro')
    return {'f1': f1}

# Training arguments with validation
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    logging_steps=50,
    logging_dir='./logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    seed=42,
    report_to='none'
)

# Trainer with validation and early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Fine-tune with validation
trainer.train()

# === Inference Function ===
def predict(csv_path):
    """
    Perform multi-label emotion prediction on input CSV.
    Uses the best model from training.

    Args:
        csv_path (str): Path to CSV with a 'text' column.
    Returns:
        np.ndarray: Binary array of shape (n_samples, 5) for [anger, fear, joy, sadness, surprise].
    """
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].fillna('')

    # Prepare test dataset
    hf_test = HFDataset.from_pandas(df)
    hf_test = hf_test.map(tokenize, batched=True)
    hf_test.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'token_type_ids']  # Fixed columns
    )

    # Get raw predictions using best model
    outputs = trainer.predict(hf_test)
    logits = outputs.predictions

    # Convert logits to probabilities and apply threshold
    probs = sigmoid(torch.tensor(logits)).numpy()
    binary_preds = (probs >= 0.5).astype(int)

    return binary_preds

# Example usage
if __name__ == '__main__':
    # Test prediction
    #preds = predict('test.csv')
    #print("Predictions shape:", preds.shape)
    #print("Sample predictions:\n", preds[:3])