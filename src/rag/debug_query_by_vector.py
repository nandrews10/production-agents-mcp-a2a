import chromadb

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chroma_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
)

query = "incident response"

print("Loading embedding model...")

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Creating query embedding...")

query_embedding = embedding_function([query])[0]

print("Query embedding created.")
print(f"Embedding length: {len(query_embedding)}")

print("Opening Chroma...")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(
    name=COLLECTION_NAME,
)

print("Running Chroma vector query...")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
)

print("Query finished.")
print()

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, document in enumerate(documents):
    print("-" * 60)
    print(f"Rank: {i + 1}")
    print(f"Source: {metadatas[i]['source']}")
    print(f"Chunk: {metadatas[i]['chunk']}")
    print(f"Distance: {distances[i]}")
    print(document[:500])
    print()