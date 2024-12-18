import json
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

BASE_PATH = "multiwoz/db"
DB_DIR = BASE_PATH

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 


def load_db(domain: str) -> List[dict]:
    """
    Load the database for a given domain.
    
    Parameters:
    - domain (str): The domain of the database."""
    db_file = os.path.join(DB_DIR, f"{domain}_db.json")
    with open(db_file, 'r', encoding='utf-8') as f:
        db = json.load(f)
    return db


def convert_db_to_text(db: List[dict], domain: str) -> List[str]:
    """
    Convert the database entries to text format. This is required to create an embedding index.
    
    Parameters:
    - db (List[dict]): The database entries.
    - domain (str): The domain of the database.
    """
    db_texts = []
    for entity in db:
        entity_text = f"Domain: {domain}, " + ", ".join(
            [f"{key}: {value}" for key, value in entity.items() if key != 'location'])
        db_texts.append(entity_text)
    return db_texts


def prepare_db_indices(domains: List[str], save_dir: str = "models/retrieval"):
    """
    Prepare FAISS indices for the database entries of the given domains.
    
    Parameters:
    - domains (List[str]): List of domains.
    - save_dir (str): Directory to save the indices.
    """
    os.makedirs(save_dir, exist_ok=True)
    for domain in domains:
        print(f"Processing domain: {domain}")
        db = load_db(domain)
        db_texts = convert_db_to_text(db, domain)
        embeddings = embedding_model.encode(
            db_texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(
            save_dir, f"{domain}_faiss.index"))
        with open(os.path.join(save_dir, f"{domain}_db_texts.pkl"), 'wb') as f:
            pickle.dump(db_texts, f)
        print(f"Saved FAISS index and DB texts for domain: {domain}")


def load_retrieval_resources(domain: str, save_dir: str = "models/retrieval") -> Tuple[faiss.Index, List[str]]:
    """
    Load the FAISS index and database texts for a given domain.
    
    Parameters:
    - domain (str): The domain of interest.
    - save_dir (str): Directory where the indices are saved.
    """
    index_path = os.path.join(save_dir, f"{domain}_faiss.index")
    texts_path = os.path.join(save_dir, f"{domain}_db_texts.pkl")
    index = faiss.read_index(index_path)
    with open(texts_path, 'rb') as f:
        db_texts = pickle.load(f)
    return index, db_texts

def _parse_text_to_json(text: str) -> dict:
    result = {}
    parts = text.split(". ")
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            key = key.lower().strip()
            value = value.strip()
            result[key] = value
            
    return json.dumps(result)
    

def retrieve_db_entries(query: str, domain: str, top_k: int = 3, save_dir: str = "models/retrieval") -> List[str]:
    """
    Retrieve the top-k database entries for a given query and domain based on semantic similarity.
    
    Parameters:
    - query (str): The user query.
    - domain (str): The domain of interest.
    - top_k (int): Number of top entries to retrieve.
    - save_dir (str): Directory where the indices are saved.
    """
    index, db_texts = load_retrieval_resources(domain, save_dir)
    embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    similarities, indices = index.search(embedding, top_k)
    retrieved = [_parse_text_to_json(db_texts[idx]) for idx in indices[0]]
    return retrieved



if __name__ == "__main__":
    # Define domains
    domains = ["hotel", "train", "attraction",
               "restaurant", "hospital", "taxi", "bus", "police"]
    # Prepare indices (Run only once)
    prepare_db_indices(domains)

    # # Example retrieval
    retrieved = retrieve_db_entries("I need an expensive chinese restaurant in the center, there will be a table for 4 people", "restaurant", top_k=5)
    print(retrieved)
