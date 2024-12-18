# Multi-domain chatbot trained on MultiWOZ dataset

The project requires some pre-trained models to be prepared before running GUI or CLI interface.

You need:

- Pre-train a domain classifier: `python domain_classifier.py`
- Build embeddings with: `python embedding_builder.py`
- Build FAISS db indices with: `python retrieve.py`
- Pre-train message generator with: `python3 generator.py`