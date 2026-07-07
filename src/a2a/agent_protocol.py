"""
agent_protocol.py

This file defines the tiny message format that our agents use
to talk to each other.

Why this matters:

In a production agent system, agents should NOT pass random
Python dictionaries around with inconsistent keys.

Bad:

    {
        "stuff": "...",
        "x": "...",
        "maybe_user": "..."
    }

Good:

    AgentMessage(
        sender="coordinator",
        receiver="retrieval_agent",
        task="retrieve_evidence",
        payload={...},
    )

This gives us a consistent Agent-to-Agent, or A2A, protocol.

Architecture:

    Coordinator Agent
          |
          | AgentMessage
          v
    Retrieval Agent
          |
          | AgentMessage
          v
    Other future agents:
        - Guardrail Agent
        - Audit Agent
        - Cost Agent
        - Tenant Policy Agent
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from uuid import uuid4
from datetime import datetime, timezone


@dataclass
class AgentMessage:
    """
    Standard message object passed between agents.

    Fields:

    message_id:
        Unique ID for tracing one message.

    sender:
        Which agent created this message.

    receiver:
        Which agent should handle this message.

    task:
        What the receiver should do.

    payload:
        The actual business data.

    created_at:
        Timestamp for audit logging and observability.
    """

    sender: str
    receiver: str
    task: str
    payload: Dict[str, Any]

    message_id: str = field(default_factory=lambda: str(uuid4()))

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def create_message(
    sender: str,
    receiver: str,
    task: str,
    payload: Dict[str, Any],
) -> AgentMessage:
    """
    Convenience helper for creating an AgentMessage.

    This keeps message creation consistent across the project.
    """

    return AgentMessage(
        sender=sender,
        receiver=receiver,
        task=task,
        payload=payload,
    )