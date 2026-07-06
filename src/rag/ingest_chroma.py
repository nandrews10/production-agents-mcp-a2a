"""
Build a persistent Chroma vector database.

This script:

1. Walks through every document
2. Extracts text
3. Splits into chunks
4. Creates embeddings
5. Stores everything inside Chroma

This is NOT an MCP server.

It is an offline indexing job that may run once a day,
once an hour, or after new documents arrive.
"""
#Below we did just to check paths and locations etc., before moving to persistent Chroma
# from pathlib import Path

# import chromadb

# from chromadb.utils.embedding_functions import (
#     SentenceTransformerEmbeddingFunction,
# )

# from .chroma_config import (
#     CHROMA_DIR,
#     COLLECTION_NAME,
# )

# print("=" * 60)
# print("Enterprise Chroma Ingestion")
# print("=" * 60)

# print()

# print(f"Vector database location : {CHROMA_DIR}")
# print(f"Collection name          : {COLLECTION_NAME}")
"""
Build a persistent Chroma vector database.

Baby step:
- Create/reopen Chroma database
- Create/reopen collection
- Print current document count
"""

import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from src.mcp_servers.document_mcp_server import (
    collect_documents,
    extract_text_from_file,
    resolve_document_path,
)

from .chroma_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)



def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split long text into overlapping chunks.

    Example:
    chunk_size=1000, chunk_overlap=150

    Chunk 1: characters 0-1000
    Chunk 2: characters 850-1850
    Chunk 3: characters 1700-2700

    The overlap helps preserve context across chunk boundaries.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = start + chunk_size - chunk_overlap

    return chunks




print("=" * 60)
print("Enterprise Chroma Ingestion")
print("=" * 60)
print()

print(f"Vector database location : {CHROMA_DIR}")
print(f"Collection name          : {COLLECTION_NAME}")
print()

# Chroma needs a string path, not a Path object.
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Local embedding model.
# This downloads the model first time only.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection if it does not exist.
# Reopen it if it already exists.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

print("Chroma client created.")
print("Collection ready.")
print(f"Current collection count: {collection.count()}")
print()

# print("Removing existing vectors...")

# # During development we rebuild from scratch.
# collection.delete(where={})

# print("Collection is now empty.")
print("Removing existing collection if it exists...")

try:
    client.delete_collection(COLLECTION_NAME)
    print("Old collection deleted.")

except Exception:
    print("Collection did not exist yet.")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

print("Fresh collection created.")
print()

print()
print()
print("=" * 60)
print("Scanning enterprise corpus...")
print("=" * 60)

documents = collect_documents()

print(f"Found {len(documents)} documents.\n")

for doc in documents:
    print(doc["relative_path"])

print()
print("=" * 60)
print("Extracting text...")
print("=" * 60)

for doc in documents:

    relative_path = doc["relative_path"]

    print(f"\nProcessing: {relative_path}")

    try:

        path = resolve_document_path(relative_path)

        text = extract_text_from_file(
            path,
            max_pages=5,
        )

        if text and text.strip():

            # print(f"SUCCESS")
            # print(f"Characters extracted: {len(text):,}")
            chunks = split_text(
                text=text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            
            print("SUCCESS")
            print(f"Characters extracted: {len(text):,}")
            print(f"Chunks created      : {len(chunks):,}")
#             
            ids = []
            chunk_documents = []
            metadatas = []

            for chunk_number, chunk in enumerate(chunks):

                chunk_id = f"{relative_path}-{chunk_number}"

                ids.append(chunk_id)
                chunk_documents.append(chunk)

                metadatas.append(
                    {
                        "source": relative_path,
                        "chunk": chunk_number,
                        "category": doc["category"],
                    }
            )

            collection.add(
                ids=ids,
                documents=chunk_documents,
                metadatas=metadatas,
    )

            print(f"Stored {len(chunks)} chunks.")


        else:

            print("EMPTY DOCUMENT")

    except Exception as e:

        print(f"FAILED")
        print(e)
        
print()
print("=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)

print(f"Final collection size: {collection.count():,}")