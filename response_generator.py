import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch
from typing import Dict, List, Any
import os


class ResponseGenerator:
    def __init__(self, model_path: str = "./t5_finetuned"):
        """
        Initialize the ResponseGenerator with a pre-trained or fine-tuned T5 model.

        Parameters:
        - model_path (str): Path to the fine-tuned T5 model directory.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path {
                    model_path} does not exist. Please train the generator first."
            )

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def generate_response(
        self,
        user_query: str,
        domain: str,
        intents: List[str],
        slots: Dict[str, List[str]],
        db_results: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        max_length: int = 150
    ) -> str:
        """
        Generate a response based on user input, domain, intents, slots, retrieved DB results, and conversation history.

        Parameters:
        - user_query (str): The user's input message.
        - domain (str): The detected domain/domain of interest.
        - intents (List[str]): List of detected intents.
        - slots (Dict[str, List[str]]): Extracted slot-value pairs.
        - db_results (Dict[str, Any]): Retrieved database entries relevant to the query.
        - conversation_history (List[Dict[str, str]]): List of previous user and bot exchanges.
        - max_length (int): Maximum length of the generated response.

        Returns:
        - response (str): The generated system response.
        """
        try:
            # Format intents
            intents_str = "|".join(intents) if intents else "none"

            # Format slots
            slots_str = ", ".join(
                [f"{k}:{'|'.join(v)}" for k, v in slots.items()]
            ) if slots else "none"

            # Format DB results
            print("[DEBUG] DB Results:", db_results)
            if db_results:
                # Flatten db_results if values are lists
                db_str = " | ".join(db_results[domain])
            else:
                db_str = "none"
            
            # PROMPT
            input_text = (
                f"Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n"
                "### Instruction: "
                f"{user_query}"

                "### Input: "
                f"Domain: {domain}\n"
                f"Intents: {intents_str}\n"
                f"Slots: {slots_str}\n"
                "\n"
                "### Output:")

            print("[DEBUG] T5 input:", input_text)

            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            # Decode the response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            # Clean up the response if necessary
            response = response.split(
                "System:")[-1].strip() if "System:" in response else response.strip()
            return response
        except Exception as e:
            print(f"Error in generating response: {e}")
            return "I'm sorry, I encountered an error while trying to process your request."


def load_generator(model_path: str = "./t5_finetuned") -> ResponseGenerator:
    """
    Load the fine-tuned T5 model and tokenizer.

    Parameters:
    - model_path (str): Path to the fine-tuned T5 model directory.

    Returns:
    - ResponseGenerator: An instance of the ResponseGenerator class.
    """
    return ResponseGenerator(model_path=model_path)


def generate_response_t5(
    user_query: str,
    domain: str,
    intents: List[str],
    slots: Dict[str, List[str]],
    db_results: Dict[str, Any],
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Generate a response using the fine-tuned T5 model.

    Parameters:
    - user_query (str): The user's input message.
    - domain (str): The detected domain(s) from the domain classifier.
    - intents (List[str]): List of detected intents.
    - slots (Dict[str, List[str]]): Extracted slot-value pairs.
    - db_results (Dict[str, Any]): Retrieved database entries relevant to the query.
    - conversation_history (List[Dict[str, str]]): List of previous user and bot exchanges.

    Returns:
    - response (str): The generated system response.
    """
    try:
        # Initialize the generator once (avoid reloading every time)
        if not hasattr(generate_response_t5, "generator"):
            generate_response_t5.generator = ResponseGenerator(
                model_path="./t5_finetuned"
            )

        generator = generate_response_t5.generator

        # Generate response
        response = generator.generate_response(
            user_query=user_query,
            domain=domain,
            intents=intents,
            slots=slots,
            db_results=db_results,
            conversation_history=conversation_history
        )
        return response
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "I'm sorry, I encountered an error while trying to process your request."
