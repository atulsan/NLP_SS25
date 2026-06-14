# Multi-Label Emotion Detection — SemEval 2025 Task 11 (Track A)

Fine-tuned transformer model that detects **multiple co-occurring emotions** in a
text snippet. Built for [SemEval 2025 Shared Task 11](https://github.com/emotion-analysis-project/SemEval2025-task11),
Track A (multi-label emotion detection). Achieves a **macro-F1 of 0.77** on the
held-out evaluation set.

> Course project, NLP (Summer Semester 2025), Philipps-Universität Marburg.

## Problem

Given a piece of text, predict which emotions are present. Unlike single-label
classification, each sample can carry several emotions at once (e.g. *joy* **and**
*surprise*), so the model outputs an independent probability per emotion and a
per-label decision threshold is tuned to maximise F1.

## Approach

- **Model** — a pre-trained transformer encoder fine-tuned for multi-label
  classification (sigmoid output head, binary cross-entropy loss) using the
  Hugging Face `Trainer` API.
- **Threshold optimisation** — per-label decision thresholds are learned on the
  validation set and stored in `thresholds.pt`, rather than using a flat 0.5
  cut-off. This is what lifts macro-F1 on the rarer emotion labels.
- **Training** — 10 epochs with evaluation each epoch; the best checkpoint is
  selected by validation F1.

## Results

| Metric | Score |
| --- | --- |
| Macro-F1 (eval) | **0.774** |
| Eval loss | 0.855 |
| Epochs | 10 |

## Repository layout

```
main.py                  # train / predict entry point (CLI)
requirements.txt         # dependencies
thresholds.pt            # tuned per-label decision thresholds
track-a.csv              # labelled training/eval data
track-a-no-emotions.csv  # unlabelled input template for prediction
predictions.csv          # sample model output
```

## Usage

1. Download the trained model and unzip it into the project root so a
   `best_model/` folder exists:
   <https://hessenbox.uni-marburg.de/getlink/fiKPpZCajLTYqUkxRyiB5n/best_model.zip>

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train (optional — skip if you downloaded `best_model/`):

   ```bash
   python main.py train
   ```

4. Predict on your own file (replace `input_file.csv` with your test set):

   ```bash
   python main.py --input_csv input_file.csv predict
   ```

   This writes a `predictions.csv` ready for evaluation.

## Tech stack

Python · PyTorch · Hugging Face Transformers · pandas · scikit-learn

## Team

Pooja Negi, Atul Santhosh, Yadhukrishnan Pandiyatt, Vimal Kottamulla Valappil,
Tony Sebastian.

---
*Note: replace the encoder name above (e.g. RoBERTa / BERT / DeBERTa) with the
exact model used in `main.py` for full accuracy.*
