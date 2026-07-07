"""
coordinator_agent.py

The CoordinatorAgent is our supervisor agent.

Its job is NOT to retrieve documents itself.

Its job is to decide:

    1. What task needs to be done?
    2. Which specialist agent should handle it?
    3. How should the result be returned?

For now, we only have one specialist:

    RetrievalAgent

Later, the coordinator will orchestrate:

    - RetrievalAgent
    - GuardrailAgent
    - AuditAgent
    - TenantPolicyAgent
    - CostTrackingAgent
    - AnswerAgent

Current architecture:

    User request
        |
        v
    CoordinatorAgent
        |
        v
    RetrievalAgent
        |
        v
    Reranked evidence chunks
"""
import time
from uuid import uuid4
from typing import Any, Dict

from src.a2a.agent_protocol import create_message
from src.agents.base_agent import BaseAgent
from src.agents.retrieval_agent import RetrievalAgent

from src.observability.audit_logger import write_audit_event

class CoordinatorAgent(BaseAgent):
    """
    Supervisor agent.

    For this baby step, it only knows how to call RetrievalAgent.

    Later it will:
        - check auth context
        - enforce tenant isolation
        - call guardrails
        - call audit logger
        - call answer generator
    """

    def __init__(self):
        super().__init__(
            name="coordinator_agent",
            description="Coordinates specialist agents for production workflows.",
        )

        self.retrieval_agent = RetrievalAgent()

    def handle_user_query(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 8,
        preview_chars: int = 800,
    ) -> Dict[str, Any]:
        """
        Handle a user query.

        Current workflow:

            query
              |
              v
            create AgentMessage
              |
              v
            RetrievalAgent.handle_message()
              |
              v
            return evidence chunks

        This method is intentionally simple first.
        """
        start_time = time.perf_counter()
        trace_id = str(uuid4())
        
        # --------------------------------------------------------
        # Temporary identity information.
        #
        # Today:
        #     Hard-coded values for development.
        #
        # Later:
        #     JWT authentication will populate these automatically.
        # --------------------------------------------------------

        user_id = "demo_user"
        tenant_id = "demo_tenant"
        
        coordinator_span_id = str(uuid4())
        write_audit_event(
            event_type="user_query_received",
            status="started",
            details={
                "trace_id": trace_id,
                "span_id": coordinator_span_id,
                "span_name": "coordinator_received_query",
                "user_id": user_id,
                "tenant_id": tenant_id,
                "agent": self.name,
                "query": query,
                "top_k": top_k,
                "candidate_k": candidate_k,
            },
        )
        retrieval_message = create_message(
            sender=self.name,
            receiver="retrieval_agent",
            task="retrieve_evidence",
            payload={
                "query": query,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "preview_chars": preview_chars,
            },
        )

        retrieval_response = self.retrieval_agent.handle_message(
            retrieval_message
        )
        
        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
        
        retrieval_span_id = str(uuid4())
        write_audit_event(
            event_type="retrieval_completed",
            status=retrieval_response.get("status", "unknown"),
            details={
                "trace_id": trace_id,
                "span_id": retrieval_span_id,
                "span_name": "retrieval_completed",
                "user_id": user_id,
                "tenant_id": tenant_id,
                "agent": self.name,
                "query": query,
                "result_count": len(retrieval_response.get("results", [])),
                "elapsed_ms": elapsed_ms,
            },
        )
        return {
            "coordinator": self.name,
            "status": retrieval_response.get("status"),
            "trace_id": trace_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query": query,
            "retrieval_response": retrieval_response,
        }

    def handle_message(
        self,
        message,
    ) -> Dict[str, Any]:
        """
        Standard A2A entry point.

        This lets another agent call the CoordinatorAgent using AgentMessage.
        """

        if message.task != "handle_user_query":
            return {
                "agent": self.name,
                "status": "error",
                "message": f"Unsupported task: {message.task}",
            }

        query = message.payload.get("query")

        if not query:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Missing required payload field: query",
            }

        return self.handle_user_query(
            query=query,
            top_k=message.payload.get("top_k", 3),
            candidate_k=message.payload.get("candidate_k", 8),
            preview_chars=message.payload.get("preview_chars", 800),
        )