"""
Tiny reranker smoke test.
"""

from src.rag.reranker import rerank_results

results = [
    {
        "preview": "Collision deductible is $500.",
        "relative_path": "doc1.pdf",
    },
    {
        "preview": "Health insurance copay is $25.",
        "relative_path": "doc2.pdf",
    },
    {
        "preview": "Incident response procedures are defined by CISA.",
        "relative_path": "doc3.pdf",
    },
]

reranked = rerank_results(
    query="collision deductible",
    results=results,
    top_k=3,
)

for item in reranked:
    print(item["relative_path"])
    print(item["reranker_score"])
    print()