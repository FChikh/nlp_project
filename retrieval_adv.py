from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from retrieval import retrieve_db_entries
from domain_classifier import DomainClassifier


class SimpleRetriever(BaseRetriever):
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return the first k documents from the list of documents"""
        a = DomainClassifier("./domain_classifier")
        data = retrieve_db_entries(query=query, domain=a.predict(query)[0], top_k=self.k)
        data = [Document(item) for item in data]
        return data

