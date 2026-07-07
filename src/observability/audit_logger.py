"""
audit_logger.py

Simple JSONL audit logger for our production agent system.

JSONL means:
    one JSON object per line

Example:

    {"event_type": "user_query", "status": "success", ...}
    {"event_type": "retrieval", "status": "success", ...}

Why audit logging matters:

In production AI systems, we need to know:

    - who asked
    - what they asked
    - which agent handled it
    - what documents were used
    - whether it succeeded
    - when it happened

Architecture:

    CoordinatorAgent
          |
          v
    Audit Logger
          |
          v
    logs/audit.jsonl

Later:
    - send logs to CloudWatch
    - add OpenTelemetry traces
    - add tenant_id
    - add user_id
    - add cost tracking
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG_FILE = LOG_DIR / "audit.jsonl"


def write_audit_event(
    event_type: str,
    status: str,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Write one audit event to logs/audit.jsonl.

    event_type:
        What happened?
        Example:
            "user_query"
            "retrieval"
            "guardrail_check"

    status:
        Example:
            "success"
            "error"

    details:
        Extra event-specific data.
    """

    LOG_DIR.mkdir(
        exist_ok=True,
    )

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "status": status,
        "details": details,
    }

    with open(
        AUDIT_LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(event, ensure_ascii=False))
        f.write("\n")

    return event