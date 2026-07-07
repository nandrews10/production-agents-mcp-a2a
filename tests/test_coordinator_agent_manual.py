"""
Manual test for CoordinatorAgent.

Run from project root:

    python -m tests.test_coordinator_agent_manual
"""

from src.agents.coordinator_agent import CoordinatorAgent


coordinator = CoordinatorAgent()

response = coordinator.handle_user_query(
    query="collision deductible",
    top_k=3,
    candidate_k=8,
    preview_chars=800,
)

print(response["status"])
print(response["query"])
print(response["trace_id"])
print(response["user_id"])
print(response["tenant_id"])


results = response["retrieval_response"]["results"]

for item in results:
    print("-" * 60)
    print(item["relative_path"])
    print("reranker_score:", item.get("reranker_score"))
    print(item["preview"][:300])