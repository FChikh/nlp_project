# src/generator.py

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import torch
from typing import Dict, List, Any
import os


def train_generator():
    """
    Fine-tune the T5 model using the preprocessed training data,
    incorporating intents, slot_values, DB results, and conversation history.

    Parameters:
    - history_length (int): Number of previous exchanges to include in the input.
    """

    # Load training and validation data
    train_df = pd.read_csv("train_preprocessed.csv")
    val_df = pd.read_csv("test_preprocessed.csv")

    def format_example(row):

        domain = eval(row['domains'])[0] if len(eval(row['domains'])) > 0 else "none"

        # Format intents
        intents = eval(row['intents'])
        intents_str = "|".join(intents) if len(intents) > 0 else "none"

        # Format slot values
        slot_values = eval(row['slot_values'])
        if slot_values:
            slots_str = ", ".join(
                [f"{k}:{'|'.join(v)}" for k, v in slot_values.items()]
            )
        else:
            slots_str = "none"
            
        # PROMPT
        input_text = (
            f"Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            "### Instruction: "
            f"{row['user_utterance']}\n"

            "### Input:"
            f"Domain: {domain}\n"
            f"Intents: {intents_str}\n"
            f"Slots: {slots_str}\n"
            "\n"
            "### Output:")
        target_text = row['system_utterance']
        return {"input_text": input_text, "target_text": target_text}

    train_data = train_df.apply(lambda row: format_example(row), axis=1)
    train_dataset = Dataset.from_pandas(pd.DataFrame(list(train_data)))

    val_data = val_df.apply(lambda row: format_example(row), axis=1)
    val_dataset = Dataset.from_pandas(pd.DataFrame(list(val_data)))

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>"
    })

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=512,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize the datasets
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["input_text", "target_text"]
    )
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["input_text", "target_text"]
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    training_args = TrainingArguments(
        output_dir="./distilgpt2_finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="epoch",
        save_total_limit=2,
        weight_decay=0.01,
        learning_rate=3e-4,
        warmup_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./distilgpt2_finetuned")
    tokenizer.save_pretrained("./distilgpt2_finetuned")
    print("Generator training complete and model saved to './distilgpt2_finetuned/'.")


if __name__ == "__main__":
    train_generator()
    pass
