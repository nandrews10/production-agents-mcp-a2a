"""
Reusable Chroma semantic retrieval.

Production lesson:
- Do NOT recreate the Chroma client, embedding model, and collection
  from scratch inside every MCP tool call.
- Load them once, cache them, and reuse them.
"""

import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from .chroma_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
)

_client = None
_embedding_function = None
_collection = None


def get_chroma_collection():
    """
    Create Chroma resources once and reuse them.

    First call:
        loads client/model/collection

    Later calls:
        reuses existing collection
    """

    global _client
    global _embedding_function
    global _collection

    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    _embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function,
    )

    return _collection


def search_chroma(query: str, top_k: int = 5) -> dict:
    """
    Search the Chroma vector database using semantic similarity.
    """

    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return results


if __name__ == "__main__":

    query = "incident response plan"

    print("=" * 60)
    print("Chroma Semantic Retrieval Test")
    print("=" * 60)
    print()

    print(f"Query: {query}")
    print()

    results = search_chroma(
        query=query,
        top_k=5,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):

        metadata = metadatas[i]
        distance = distances[i]

        print("-" * 60)
        print(f"Rank: {i + 1}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk']}")
        print(f"Distance: {distance}")
        print()
        print(document[:700])
        print()