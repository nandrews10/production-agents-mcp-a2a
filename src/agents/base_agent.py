"""
base_agent.py

This file defines the common base class for all future agents.

Why do we need a BaseAgent?

Because every production agent should have the same minimum structure:

    - name
    - description
    - handle_message()
    - logging-friendly identity

Later, specialized agents will inherit from this:

    BaseAgent
        |
        ├── RetrievalAgent
        ├── GuardrailAgent
        ├── AuditAgent
        ├── TenantPolicyAgent
        └── CostTrackingAgent

Architecture:

    AgentMessage
        |
        v
    BaseAgent.handle_message()
        |
        v
    Specialized agent logic
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.a2a.agent_protocol import AgentMessage


class BaseAgent(ABC):
    """
    Abstract base class for production agents.

    ABC means:
        Abstract Base Class.

    An abstract class is a template.

    You do NOT usually run BaseAgent directly.

    Instead, you create child classes like:

        RetrievalAgent(BaseAgent)

    and force them to implement handle_message().
    """

    def __init__(
        self,
        name: str,
        description: str,
    ):
        """
        Initialize the agent.

        name:
            Short stable agent name.
            Example:
                retrieval_agent

        description:
            Human-readable explanation of what this agent does.
        """

        self.name = name
        self.description = description

    @abstractmethod
    def handle_message(
        self,
        message: AgentMessage,
    ) -> Dict[str, Any]:
        """
        Process an incoming AgentMessage.

        Every real agent must implement this method.

        Input:
            AgentMessage from another agent.

        Output:
            Dictionary result.

        Later:
            We will add audit logging, tenant ID, trace ID,
            latency tracking, and guardrail results here.
        """

        pass

    def describe(self) -> Dict[str, str]:
        """
        Return basic metadata about this agent.

        This is useful for:
            - debugging
            - agent registries
            - future A2A discovery
            - documentation
        """

        return {
            "name": self.name,
            "description": self.description,
        }