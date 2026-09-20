"""Sandboxed coding-agent integration."""

from .contracts import AgentResult, AgentTaskPacket
from .sandbox import ContainerAgent, SandboxPolicy

__all__ = ["AgentResult", "AgentTaskPacket", "ContainerAgent", "SandboxPolicy"]
