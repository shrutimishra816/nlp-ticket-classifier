"""
bert_finetune.py — Fine-tune a real BERT model on the ticket categories
------------------------------------------------------------------------
The original classifier.py uses TF-IDF + Logistic Regression, which the
README/resume once described as "BERT fine-tuning" — that wasn't accurate.
This script actually fine-tunes distilbert-base-uncased on the same
honest, leakage-free train/test split used in classifier.py (held-out
templates per category, not a random split of augmented sentences).

REQUIRES INTERNET ACCESS to huggingface.co to download the pretrained
weights. It has NOT been run to completion in the environment that wrote
it (sandboxed, no huggingface.co access) — run it yourself and use the
real number it prints / saves to evaluation_report_bert.json. Do not
put an accuracy on a resume or README until this has actually been run.

Usage:
    pip install transformers torch datasets scikit-learn pandas numpy
    python bert_finetune.py
"""

import json
import numpy as np
from classifier import generate_train_test_split, CATEGORIES

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_NAME = "distilbert-base-uncased"
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def build_datasets():
    train_df, test_df = generate_train_test_split(
        n_per_class_train=200, n_per_class_test=40, templates_held_out_per_class=4
    )
    train_df["label"] = train_df["category"].map(LABEL2ID)
    test_df["label"] = test_df["category"].map(LABEL2ID)
    return (
        Dataset.from_pandas(train_df[["text", "label"]]),
        Dataset.from_pandas(test_df[["text", "label"]]),
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds, test_ds = build_datasets()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./bert_ticket_classifier",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal held-out-template evaluation:", metrics)

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = test_ds["label"]
    report = classification_report(
        y_true, y_pred, target_names=CATEGORIES, output_dict=True
    )

    with open("evaluation_report_bert.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "accuracy": metrics["eval_accuracy"],
                "f1_weighted": metrics["eval_f1_weighted"],
                "eval_methodology": "held-out templates per category (4 of ~18), "
                "same leakage-free split used in classifier.py",
                "per_category": report,
            },
            f,
            indent=2,
        )
    print("Saved: evaluation_report_bert.json")

    model.save_pretrained("./bert_ticket_classifier")
    tokenizer.save_pretrained("./bert_ticket_classifier")
    print("Saved fine-tuned model to ./bert_ticket_classifier")


if __name__ == "__main__":
    main()
