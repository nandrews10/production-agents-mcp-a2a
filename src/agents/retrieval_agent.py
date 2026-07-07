"""
retrieval_agent.py

This file defines the RetrievalAgent.

The RetrievalAgent is the first real specialist agent
in our production A2A system.

Its job:

    User query
        |
        v
    RetrievalAgent
        |
        v
    hybrid / semantic / reranked retrieval
        |
        v
    evidence chunks

Important:

This agent does NOT call an LLM yet.

It only retrieves evidence.

Later the CoordinatorAgent will decide:

    - call RetrievalAgent
    - call GuardrailAgent
    - call AnswerAgent
    - call AuditAgent

Architecture:

    CoordinatorAgent
          |
          | AgentMessage(task="retrieve_evidence")
          v
    RetrievalAgent
          |
          v
    reranked_hybrid_search_documents()
          |
          v
    evidence chunks
"""

from typing import Any, Dict

from src.a2a.agent_protocol import AgentMessage
from src.agents.base_agent import BaseAgent

# Reuse our MCP server retrieval function directly for now.
#
# Later we may separate pure retrieval logic away from MCP,
# but for this baby step we reuse the working function.
from src.mcp_servers.document_mcp_server import reranked_hybrid_search_documents


class RetrievalAgent(BaseAgent):
    """
    Specialist agent responsible for evidence retrieval.

    It receives an AgentMessage and returns reranked chunks.
    
    No generation just retrieval
    """

    def __init__(self):
        super().__init__(
            name="retrieval_agent",
            description="Retrieves reranked evidence chunks from the enterprise corpus.",
        )

    def handle_message(
        self,
        message: AgentMessage,
    ) -> Dict[str, Any]:
        """
        Handle a retrieval request.

        Expected message:

            task = "retrieve_evidence"

            payload = {
                "query": "...",
                "top_k": 5,
                "candidate_k": 10,
                "preview_chars": 1200
            }

        Returns:
            {
                "agent": "retrieval_agent",
                "task": "retrieve_evidence",
                "results": [...]
            }
        """

        if message.task != "retrieve_evidence":
            return {
                "agent": self.name,
                "status": "error",
                "message": f"Unsupported task: {message.task}",
            }

        query = message.payload.get("query")
        top_k = message.payload.get("top_k", 5)
        candidate_k = message.payload.get("candidate_k", 10)
        preview_chars = message.payload.get("preview_chars", 1200)

        if not query:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Missing required payload field: query",
            }

        results = reranked_hybrid_search_documents(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            preview_chars=preview_chars,
        )

        return {
            "agent": self.name,
            "status": "success",
            "task": message.task,
            "query": query,
            "results": results,
        }