import re
from pathlib import Path

import chromadb
from litellm import completion
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent

# Resolves to <project_root>/cars_vectorstore regardless of caller's cwd.
# frontier_agent.py lives at <root>/auto_pricer/agents/frontier_agent.py,
# so three .parent calls walks back up to <root>.
VECTORSTORE_PATH = str(Path(__file__).resolve().parent.parent.parent / "cars_vectorstore")
COLLECTION_NAME = "cars"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini/gemini-2.5-flash"
N_COMPARABLES = 5


class FrontierAgent(BaseAgent):
    """Wraps Gemini 2.5 Flash with RAG comparable-price injection.

    Retrieves the N most similar cars from the ChromaDB vectorstore (built
    in notebooks/7_rag.ipynb) and injects them as context into the prompt.

    Uses chain-of-thought prompting: Gemini reasons briefly about how the
    target car compares to the retrieved comps (condition, mileage, trim
    differences) before committing to a final number, rather than emitting
    a bare price. This tends to produce better-calibrated estimates than
    asking for a number with no reasoning. The response is parsed for an
    explicit "FINAL PRICE: $X" line to avoid accidentally grabbing a comp's
    price mentioned earlier in the reasoning.

    Uses temperature=0 for the least variance available, but Gemini is
    still not fully deterministic even so (~±$1-2k swing observed on
    identical inputs). `seed` was tried but litellm (1.92.0, as deployed)
    rejects it outright for Gemini models with UnsupportedParamsError —
    dropped rather than worked around, since it wasn't reliably improving
    determinism anyway.

    Cost: ~$0.0001-0.0002 per call at current Gemini 2.5 Flash pricing
    ($0.30/1M input, $2.50/1M output tokens, as of July 2026) — negligible
    at realistic usage volumes.

    vectorstore_path defaults to an absolute path resolved from this
    package's own location (not the caller's cwd) — safe to call from any
    notebook or working directory. Pass an absolute path (e.g. a mounted
    Modal Volume) to run this in a cloud/deployed context.
    """

    def __init__(self, vectorstore_path: str = VECTORSTORE_PATH):
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=vectorstore_path)
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
        message = f"""Estimate the price of this used car. Think through the comparison
step by step before giving your final answer.

Car to price:
{description}

{context}

Reason briefly about how this car compares to the listed comparables (condition,
mileage, trim, any notable differences) and how that should move the price up or
down relative to them. Then, on its own final line, give your answer in exactly
this format with no other text on that line:

FINAL PRICE: $<number>"""
        return [{"role": "user", "content": message}]

    def price(self, description: str) -> float:
        response = completion(
            model=GEMINI_MODEL,
            messages=self._messages_for(description),
            temperature=0,
        )
        text = response.choices[0].message.content

        # Primary: parse the explicit "FINAL PRICE: $X" line, so we don't
        # accidentally grab a comp's price mentioned earlier in the reasoning.
        final_match = re.search(r"FINAL PRICE:\s*\$?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if final_match:
            return float(final_match.group(1).replace(",", ""))

        # Fallback: if the model didn't follow the format, take the LAST
        # number in the response rather than the first — reasoning tends to
        # mention comp prices early and the actual answer late.
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
        return float(numbers[-1]) if numbers else 0.0