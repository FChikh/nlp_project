from schema_parser import SchemaParser
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np

class EmbeddingBuilder:
    def __init__(self, schema_parser: SchemaParser, model_name: str = 'all-MiniLM-L6-v2'):
        self.schema_parser = schema_parser
        self.model = SentenceTransformer(model_name)
        self.intent_embeddings = {}  # {service: {intent: embedding}}
        self.slot_embeddings = {}    # {service: {slot: embedding}}
        self.intent_faiss_indices = {}  # {service: faiss.Index}
        self.slot_faiss_indices = {}    # {service: faiss.Index}

    def build_embeddings(self):
        services = self.schema_parser.get_services()
        for service in services:
            print(f"\nProcessing service: {service}")
            intents = self.schema_parser.get_intents(service)
            slots = self.schema_parser.get_slots(service)

            # Embed intents
            if not intents:
                print(
                    f"Warning: No intents found for service '{service}'. Skipping intents embedding.")
                continue  # Skip to next service

            intent_texts = [intent['description'] for intent in intents]
            intent_names = [intent['name'] for intent in intents]
            try:
                intent_embeddings = self.model.encode(
                    intent_texts, convert_to_numpy=True)
                print(
                    f"Generated {len(intent_embeddings)} intent embeddings for service '{service}'.")
            except Exception as e:
                print(f"Error encoding intents for service '{service}': {e}")
                intent_embeddings = np.array([])  # Assign empty array

            if len(intent_embeddings) == 0:
                print(
                    f"Warning: No intent embeddings generated for service '{service}'.")
                continue  

            self.intent_embeddings[service] = dict(
                zip(intent_names, intent_embeddings))

            if not slots:
                print(
                    f"Warning: No slots found for service '{service}'. Skipping slots embedding.")
                continue  # Skip to next service

            slot_texts = []
            slot_names = []
            for slot in slots:
                slot_name = slot['name']
                slot_description = slot['description']
                if slot.get('is_categorical', False) and slot.get('possible_values'):
                    # Combine slot description with possible values
                    for value in slot['possible_values']:
                        combined_text = f"{slot_description}. Possible value: {value}"
                        slot_texts.append(combined_text)
                        # Unique identifier
                        slot_names.append(f"{slot_name}:{value}")
                else:
                    # Non-categorical slots
                    slot_texts.append(slot_description)
                    slot_names.append(slot_name)

            try:
                slot_embeddings = self.model.encode(
                    slot_texts, convert_to_numpy=True)
                print(
                    f"Generated {len(slot_embeddings)} slot embeddings for service '{service}'.")
            except Exception as e:
                print(f"Error encoding slots for service '{service}': {e}")
                slot_embeddings = np.array([])  # Assign empty array

            if len(slot_embeddings) == 0:
                print(
                    f"Warning: No slot embeddings generated for service '{service}'.")
                continue  # Skip if no embeddings

            self.slot_embeddings[service] = dict(
                zip(slot_names, slot_embeddings))


    def build_faiss_indices(self, save_dir: str = "models/embeddings"):
        os.makedirs(save_dir, exist_ok=True)
        for service in self.schema_parser.get_services():
            # Build FAISS index for intents
            intents = list(self.intent_embeddings[service].keys())
            intent_vectors = self.intent_embeddings[service].values()
            intent_matrix = list(self.intent_embeddings[service].values())
            if not intent_matrix:
                continue
            intent_matrix = np.array(intent_matrix).astype('float32')
            faiss.normalize_L2(intent_matrix)
            dimension = intent_matrix.shape[1]
            intent_index = faiss.IndexFlatIP(dimension)
            intent_index.add(intent_matrix)
            self.intent_faiss_indices[service] = intent_index
            # Save the index and mapping
            faiss.write_index(intent_index, os.path.join(
                save_dir, f"{service}_intent_faiss.index"))
            with open(os.path.join(save_dir, f"{service}_intent_mapping.pkl"), 'wb') as f:
                pickle.dump(intents, f)

            # Build FAISS index for slots
            slots = list(self.slot_embeddings[service].keys())
            slot_vectors = self.slot_embeddings[service].values()
            print(slot_vectors)
            slot_matrix = list(self.slot_embeddings[service].values())
            if not slot_matrix:
                continue
            slot_matrix = np.array(slot_matrix).astype('float32')
            faiss.normalize_L2(slot_matrix)
            slot_index = faiss.IndexFlatIP(dimension)
            slot_index.add(slot_matrix)
            self.slot_faiss_indices[service] = slot_index
            # Save the index and mapping
            faiss.write_index(slot_index, os.path.join(
                save_dir, f"{service}_slot_faiss.index"))
            with open(os.path.join(save_dir, f"{service}_slot_mapping.pkl"), 'wb') as f:
                pickle.dump(slots, f)

    def save_embeddings(self, save_dir: str = "models/embeddings"):
        self.build_faiss_indices(save_dir=save_dir)
        print(f"Embeddings and FAISS indices saved to {save_dir}")

if __name__ == "__main__":
    schema_parser = SchemaParser("multiwoz/data/MultiWOZ_2.2/schema.json")
    embedding_builder = EmbeddingBuilder(schema_parser, model_name='fine_tune/fine-tuned-intent-bert')
    embedding_builder.build_embeddings()
    embedding_builder.build_faiss_indices()
    embedding_builder.save_embeddings()