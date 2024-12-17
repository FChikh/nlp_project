# src/intents_slots_extractor.py

from schema_parser import SchemaParser
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
from typing import Tuple, List, Dict, Any
from domain_classifier import DomainClassifier


class IntentsSlotsExtractor:
    def __init__(
        self,
        schema_parser: SchemaParser,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        embeddings_dir: str = "models/embeddings",
        intent_threshold: float = 0.3,
        categorical_slot_threshold: float = 0.4,
        non_categorical_slot_threshold: float = 0.4
    ):
        """
        Initialize the IntentsSlotsExtractor with SchemaParser and embedding resources.

        Parameters:
        - schema_parser (SchemaParser): Instance of SchemaParser.
        - embedding_model_name (str): Name of the pre-trained SentenceTransformer model.
        - embeddings_dir (str): Directory where FAISS indices and mappings are stored.
        - intent_threshold (float): Similarity threshold for intents.
        - categorical_slot_threshold (float): Similarity threshold for categorical slots.
        - non_categorical_slot_threshold (float): Similarity threshold for non-categorical slots.
        """
        self.schema_parser = schema_parser
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings_dir = embeddings_dir
        self.intent_threshold = intent_threshold
        self.categorical_slot_threshold = categorical_slot_threshold
        self.non_categorical_slot_threshold = non_categorical_slot_threshold

        # Initialize FAISS indices and mappings
        self.intent_faiss_indices = {}
        self.intent_mappings = {}
        self.slot_faiss_indices = {}
        self.slot_mappings = {}
        self.slot_types = {}  # Mapping: service -> slot -> slot_type
        self.load_faiss_indices()

    def load_faiss_indices(self):
        services = self.schema_parser.get_services()
        for service in services:
            # Load FAISS index for intents
            intent_index_path = os.path.join(
                self.embeddings_dir, f"{service}_intent_faiss.index")
            intent_mapping_path = os.path.join(
                self.embeddings_dir, f"{service}_intent_mapping.pkl")
            if os.path.exists(intent_index_path) and os.path.exists(intent_mapping_path):
                index = faiss.read_index(intent_index_path)
                with open(intent_mapping_path, 'rb') as f:
                    intents = pickle.load(f)
                self.intent_faiss_indices[service] = index
                self.intent_mappings[service] = intents

            # Load FAISS index for slots
            slot_index_path = os.path.join(
                self.embeddings_dir, f"{service}_slot_faiss.index")
            slot_mapping_path = os.path.join(
                self.embeddings_dir, f"{service}_slot_mapping.pkl")
            if os.path.exists(slot_index_path) and os.path.exists(slot_mapping_path):
                index = faiss.read_index(slot_index_path)
                with open(slot_mapping_path, 'rb') as f:
                    slots = pickle.load(f)
                self.slot_faiss_indices[service] = index
                self.slot_mappings[service] = slots

    def extract_intent(self, user_query: str, service: str, top_k: int = 1) -> Tuple[str, float]:
        """
        Extract the most similar intent for a given user query and service.

        Parameters:
        - user_query (str): The user's input message.
        - service (str): The service/domain of interest.
        - top_k (int): Number of top intents to retrieve.

        Returns:
        - Tuple[str, float]: The most probable intent and its similarity score, or ("none", 0.0).
        """
        if service not in self.intent_faiss_indices:
            return "none", 0.0
        query_embedding = self.model.encode(
            [user_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.intent_faiss_indices[service].search(
            query_embedding, top_k)
        if distances.size == 0:
            return "none", 0.0
        best_distance = distances[0][0]
        best_index = indices[0][0]
        if best_distance >= self.intent_threshold:
            return self.intent_mappings[service][best_index], best_distance
        else:
            return "none", 0.0

    def extract_slots(self, user_query: str, service: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract the most similar slots for a given user query and service.

        Parameters:
        - user_query (str): The user's input message.
        - service (str): The service/domain of interest.
        - top_k (int): Number of top slots to retrieve.

        Returns:
        - List[Tuple[str, float]]: List of tuples containing slot names and their similarity scores.
        """
        if service not in self.slot_faiss_indices:
            return []
        query_embedding = self.model.encode(
            [user_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.slot_faiss_indices[service].search(
            query_embedding, top_k)
        slots = []
        for distance, idx in zip(distances[0], indices[0]):
            slot_name = self.slot_mappings[service][idx]
            is_slot_categorial = ':' in slot_name
            threshold = self.categorical_slot_threshold if is_slot_categorial else self.non_categorical_slot_threshold 
            if distance >= threshold:
                slots.append((slot_name, distance))
        return slots

    def extract_intents_slots(self, user_query: str, services: List[str], top_k_intent: int = 1, top_k_slot: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Extract intents and slots across multiple services.

        Parameters:
        - user_query (str): The user's input message.
        - services (List[str]): List of services/domains to consider.
        - top_k_intent (int): Number of top intents to retrieve per service.
        - top_k_slot (int): Number of top slots to retrieve per service.

        Returns:
        - Dict[str, Dict[str, Any]]: Mapping of service to extracted intent and slots.
        """
        results = {}
        for service in services:
            # Extract intents
            intents = []
            for _ in range(top_k_intent):
                intent, score = self.extract_intent(
                    user_query, service, top_k=1)
                if intent != "none":
                    intents.append((intent, score))
            # If multiple intents exceed the threshold, select the one with highest score
            if len(intents) > 1:
                intents = sorted(intents, key=lambda x: x[1], reverse=True)[:1]
            elif len(intents) == 0:
                intents = [("none", 0.0)]
            # Extract slots
            slots = self.extract_slots(user_query, service, top_k=top_k_slot)
            
            # For categorical slots, select only the one with highest score
            categorical_slots = [
                slot for slot in slots if ':' in slot[0]]
            non_categorical_slots = [slot for slot in slots if not ':' in slot[0]]
            if len(categorical_slots) > 1:
                # Select the categorical slot with the highest similarity score
                categorical_slots = [
                    max(categorical_slots, key=lambda x: x[1])]
            # Combine slots
            final_slots = [slot[0]
                           for slot in categorical_slots + non_categorical_slots]
            if intents[0][0] != "none" or final_slots:
                results[service] = {
                    "intent": intents[0][0],
                    "slots": final_slots
                }
        return results


if __name__ == "__main__":
    schema_path = "multiwoz/data/MultiWOZ_2.2/schema.json"  # Update with actual path
    schema_parser = SchemaParser(schema_path)
    extractor = IntentsSlotsExtractor(schema_parser)
    domain_cls = DomainClassifier('./domain_classifier')

    user_query = "I'm going to stansted airport tuesday at 17:30, i would like to book a train"
    print("Query:", user_query)
    print("===============================================")
    domain = domain_cls.predict(user_query)
    print('Domain:', domain)
    services = domain

    extracted = extractor.extract_intents_slots(user_query, services)
    print(extracted)