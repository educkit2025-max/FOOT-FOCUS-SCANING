"""Shared contract for every foot-scan agent."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentOutput:
    """Structured agent result.

    `data` is JSON-serializable and is the only part other agents may read.
    `artifacts` holds in-memory arrays (masks, homographies) and is never serialized.
    """

    agent: str
    ok: bool
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    messages: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "ok": self.ok,
            "confidence": round(float(self.confidence), 3),
            "data": self.data,
            "error": self.error,
            "messages": list(self.messages),
        }

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_dict().items() if k != "data"}

    def readonly_data(self) -> dict[str, Any]:
        """Deep copy handed downstream so no agent can mutate another agent's output."""
        return copy.deepcopy(self.data)


def failed(agent: str, error: str, messages: list[str] | None = None, data: dict[str, Any] | None = None) -> AgentOutput:
    return AgentOutput(agent=agent, ok=False, confidence=0.0, error=error, messages=messages or [], data=data or {})
