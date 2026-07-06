"""
rerank_cli.py

Command-line wrapper for the BGE reranker.

This script is launched by the MCP server as a child
Python process.

Flow

MCP
 |
 +--> writes candidates.json
 |
 +--> launches this script
 |
 +--> this script reads candidates.json
 |
 +--> calls reranker.py
 |
 +--> writes reranked_results.json
 |
 +--> MCP reads reranked_results.json
"""

import argparse
import json

from src.rag.reranker import rerank_results


def main():
    """
    Entry point.

    Responsibilities

    1. Read command-line arguments.
    2. Load candidate chunks from JSON.
    3. Call the BGE reranker.
    4. Save reranked results as JSON.
    """

    parser = argparse.ArgumentParser(
        description="Rerank candidate chunks using the BGE CrossEncoder."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Original user query.",
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="JSON file containing candidate chunks.",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Where the reranked JSON should be written.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final reranked chunks.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Read candidate chunks produced by hybrid retrieval.
    # ------------------------------------------------------------

    with open(
        args.input_file,
        "r",
        encoding="utf-8",
    ) as f:

        candidates = json.load(f)

    # ------------------------------------------------------------
    # Call the production reranker.
    # ------------------------------------------------------------

    reranked = rerank_results(
        query=args.query,
        results=candidates,
        top_k=args.top_k,
    )

    # ------------------------------------------------------------
    # Save reranked results.
    # ------------------------------------------------------------

    with open(
        args.output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            reranked,
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()