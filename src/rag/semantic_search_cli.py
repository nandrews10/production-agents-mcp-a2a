

"""
semantic_search_cli.py

Command-line semantic search wrapper.

This script is called from PowerShell or from the MCP server
as a separate child Python process.

Why does this file exist?

We discovered that this works reliably:

    1. Create query embedding manually.
    2. Query Chroma using query_embeddings=[...].

But this was unreliable inside MCP subprocess:

    collection.query(query_texts=[query])

So this CLI intentionally avoids query_texts.
It uses explicit query embeddings instead.

PowerShell example:

    python -m src.rag.semantic_search_cli --query "incident response" --top-k 5

Flow:

    PowerShell / MCP
          |
          v
    semantic_search_cli.py
          |
          v
    parse command-line arguments
          |
          v
    create embedding for query
          |
          v
    open Chroma collection
          |
          v
    query by vector
          |
          v
    print JSON results
"""

import argparse
import json

import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from dotenv import parser

from src.rag.chroma_config import (
    CHROMA_DIR,
    COLLECTION_NAME,
)


def main():
    """
    Main entry point.

    This function runs when we call:

        python -m src.rag.semantic_search_cli ...

    The job of main() is:

    1. Read command-line arguments.
    2. Run semantic retrieval.
    3. Print JSON to stdout.
    """

    parser = argparse.ArgumentParser(
        description="Run semantic search against the local Chroma vectorstore."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="User question or search query. Example: incident response",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    parser.add_argument(
    "--preview-chars",
    type=int,
    default=800,
    help="Maximum characters to return per result preview.",
    )

    parser.add_argument(
    "--output-file",
    required=False,
    help="Optional JSON output file.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. Load embedding model.
    # ------------------------------------------------------------
    # This model converts text into a 384-dimensional vector.
    # We use the same model that was used during ingestion.
    # This must match ingest_chroma.py.
    # ------------------------------------------------------------

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # ------------------------------------------------------------
    # 2. Convert the user's query into a vector.
    # ------------------------------------------------------------
    # embedding_function expects a list of strings.
    # So we pass [args.query].
    #
    # It returns a list of embeddings.
    # Since we only passed one query, we take [0].
    # ------------------------------------------------------------

    query_embedding = embedding_function([args.query])[0]

    # ------------------------------------------------------------
    # 3. Open the persistent Chroma database.
    # ------------------------------------------------------------
    # This database was created by:
    #
    #     python -m src.rag.ingest_chroma
    #
    # It contains 168 chunks right now.
    # ------------------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # ------------------------------------------------------------
    # 4. Query Chroma by vector.
    # ------------------------------------------------------------
    # Important:
    # We use query_embeddings, NOT query_texts.
    #
    # query_texts would make Chroma call the embedding model internally.
    # That worked outside MCP but caused hangs in the MCP subprocess path.
    # Explicit query_embeddings is more predictable.
    # ------------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
    )

    # ------------------------------------------------------------
    # 5. Convert Chroma's nested result format into clean JSON.
    # ------------------------------------------------------------

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for i, document in enumerate(documents):
        metadata = metadatas[i]

        output.append(
            {
                "rank": i + 1,
                "relative_path": metadata["source"],
                "category": metadata["category"],
                "chunk": metadata["chunk"],
                "distance": float(distances[i]),
                "preview": document[: args.preview_chars],
            }
        )

    # ------------------------------------------------------------
    # 6. Print JSON only.
    # ------------------------------------------------------------
    # The MCP server will capture stdout and parse this JSON.
    # ------------------------------------------------------------

    #print(json.dumps(output, ensure_ascii=False))
        # ------------------------------------------------------------
    # Convert our Python list into a JSON string.
    #
    # We may either:
    #
    # 1. Print it to stdout (normal CLI usage), OR
    # 2. Save it to a file (when called from MCP).
    # ------------------------------------------------------------

    json_output = json.dumps(
        output,
        ensure_ascii=False,
    )

    # ------------------------------------------------------------
    # If an output file was supplied on the command line,
    # write the JSON there.
    #
    # Otherwise, behave like a normal CLI program and
    # print the JSON to stdout.
    # ------------------------------------------------------------

    if args.output_file:

        with open(
            args.output_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(json_output)

    else:

        print(
            json_output,
            flush=True,
        )


if __name__ == "__main__":
    main()



# """

# This is the OLD one which we dropped.  See comments

# CLI wrapper for Chroma semantic search.

# This file is called by the MCP server using subprocess.

# Why?
# - Standalone semantic retrieval works.
# - MCP stdio hangs when running embedding/query directly.
# - So we isolate semantic retrieval in a child Python process.
# """

# import argparse
# import json

# from src.rag.retrieve_chroma import search_chroma


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--query", required=True)
#     parser.add_argument("--top-k", type=int, default=5)
#     parser.add_argument("--preview-chars", type=int, default=800)

#     args = parser.parse_args()

#     results = search_chroma(
#         query=args.query,
#         top_k=args.top_k,
#     )

#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]

#     output = []

#     for i, document in enumerate(documents):
#         metadata = metadatas[i]

#         output.append(
#             {
#                 "rank": i + 1,
#                 "relative_path": metadata["source"],
#                 "category": metadata["category"],
#                 "chunk": metadata["chunk"],
#                 "distance": float(distances[i]),
#                 "preview": document[: args.preview_chars],
#             }
#         )

#     print(json.dumps(output, ensure_ascii=False))


# if __name__ == "__main__":
#     main()