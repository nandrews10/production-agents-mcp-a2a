"""
Manual test for RetrievalAgent.

Run from project root:

    python tests/test_retrieval_agent_manual.py
"""

from src.a2a.agent_protocol import create_message
from src.agents.retrieval_agent import RetrievalAgent


agent = RetrievalAgent()

message = create_message(
    sender="coordinator_agent",
    receiver="retrieval_agent",
    task="retrieve_evidence",
    payload={
        "query": "collision deductible",
        "top_k": 3,
        "candidate_k": 8,
        "preview_chars": 800,
    },
)

response = agent.handle_message(message)

print(response["status"])
print(response["agent"])
print(response["query"])

for item in response["results"]:
    print("-" * 60)
    print(item["relative_path"])
    print(item.get("reranker_score"))
    print(item["preview"][:300])