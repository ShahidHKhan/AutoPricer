import re

import chromadb
from litellm import completion
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent

VECTORSTORE_PATH = "cars_vectorstore"
COLLECTION_NAME = "cars"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini/gemini-2.5-flash"
N_COMPARABLES = 5


class FrontierAgent(BaseAgent):
    """Wraps Gemini 2.5 Flash with RAG comparable-price injection.

    Retrieves the N most similar cars from the ChromaDB vectorstore (built
    in notebooks/7_rag.ipynb) and injects them as context into the prompt.
    Unlike SpecialistAgent, this works well with RAG since Gemini has no
    rigid trained prompt shape to violate (single test: $11,075 vs $10,990
    actual; 15-car sample MAE: $1,599.20).

    Cost: ~$0.0001-0.0002 per call at current Gemini 2.5 Flash pricing
    ($0.30/1M input, $2.50/1M output tokens, as of July 2026) — negligible
    at realistic usage volumes.
    """

    def __init__(self):
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        self.collection = client.get_or_create_collection(COLLECTION_NAME)

    def _find_similars(self, description: str, n_results: int = N_COMPARABLES):
        vector = self.encoder.encode([description]).astype(float).tolist()
        results = self.collection.query(
            query_embeddings=vector,
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        documents = results["documents"][0]
        prices = [m["price"] for m in results["metadatas"][0]]
        return documents, prices

    def _make_context(self, similar_docs, prices) -> str:
        context = "Here are some similar used car listings with their prices:\n\n"
        for doc, price in zip(similar_docs, prices):
            context += f"Price: ${price:,.0f}\n{doc}\n\n"
        return context.strip()

    def _messages_for(self, description: str):
        docs, prices = self._find_similars(description)
        context = self._make_context(docs, prices)
        message = f"""Estimate the price of this used car. Respond with the price only, no explanation.

Car to price:
{description}

{context}"""
        return [{"role": "user", "content": message}]

    def price(self, description: str) -> float:
        response = completion(
            model=GEMINI_MODEL,
            messages=self._messages_for(description),
            temperature=0,
            seed=42,
        )
        text = response.choices[0].message.content
        match = re.search(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
        return float(match.group()) if match else 0.0