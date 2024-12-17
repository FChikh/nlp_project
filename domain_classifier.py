import pandas as pd
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

from typing import List

from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class DomainClassifier:
    def __init__(self, model_path: str):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Load label mappings
        label_df = pd.read_csv(os.path.join(model_path, "domain_labels.csv"))
        self.label_list = label_df['0'].tolist()
        self.label2id = {label: idx for idx,
                         label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def predict(self, text: str) -> List[str]:
        inputs = self.tokenizer.encode(
            text, return_tensors='pt', truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits
        predicted_class_id = logits.argmax().item()
        predicted_domain = self.id2label[predicted_class_id]
        return [predicted_domain]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}


def main():
    train_df = pd.read_csv("train_preprocessed.csv")
    # Extract a primary domain. If multiple domains, pick the first in alphabetical order for simplicity

    def pick_domain(dom_str):
        dom_list = eval(dom_str)
        return dom_list[0] if len(dom_list) > 0 else "none"

    train_df['domain'] = train_df['domains'].apply(pick_domain)
    train_df = train_df[train_df['domain'] != "none"]

    val_df = pd.read_csv("test_preprocessed.csv")
    val_df['domain'] = val_df['domains'].apply(pick_domain)
    val_df = val_df[val_df['domain'] != "none"]

    label_list = sorted(train_df['domain'].unique().tolist())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased")

    def encode(examples):
        return tokenizer(examples['user_utterance'], truncation=True, padding="max_length", max_length=64)

    train_ds = Dataset.from_pandas(train_df[['user_utterance', 'domain']])
    val_ds = Dataset.from_pandas(val_df[['user_utterance', 'domain']])

    train_ds = train_ds.map(encode, batched=True)
    val_ds = val_ds.map(encode, batched=True)

    def label_map(example):
        return {'labels': label2id[example['domain']]}

    train_ds = train_ds.map(label_map)
    val_ds = val_ds.map(label_map)

    train_ds = train_ds.remove_columns(["user_utterance", "domain"])
    val_ds = val_ds.remove_columns(["user_utterance", "domain"])

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(label_list)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="./domain_classifier",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2, 
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained("./domain_classifier")
    tokenizer.save_pretrained("./domain_classifier")

    eval_results = trainer.evaluate()
    print(eval_results)

    # Save label mappings
    pd.Series(label_list).to_csv("domain_labels.csv", index=False)
    print("Domain classifier training complete.")


if __name__ == "__main__":
    main()
