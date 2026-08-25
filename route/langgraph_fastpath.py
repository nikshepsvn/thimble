"""Example: Thimble as a fast-path node in a LangGraph agent.

The pattern requested in langchain-ai/docs#4321, with the three guarantees the
maintainers asked for:

  1. high-confidence, command-shaped requests bypass the LLM entirely —
     including ARGUMENTS, not just the route: the synthetic AIMessage carries
     complete, grammar-validated tool_calls (~130ms, local, 48M params);
  2. abstention is first-class: low margin/value-confidence falls through to
     the normal agent node untouched;
  3. traceability: every fast-path message is tagged with router name, both
     confidence scores, and the thresholds that admitted it, so operators
     never attribute a dispatcher decision to the LLM.

Requires: langgraph, langchain-core. The dispatcher itself needs neither —
see dispatch.py for the dependency-free core.
"""
from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command

from .dispatch import ThimbleDispatcher

_dispatcher = ThimbleDispatcher()


def fastpath_node(state: dict) -> Command:
    query = state["messages"][-1].content
    tools = state["tool_schemas"]          # OpenAI-style function schemas
    d = _dispatcher.dispatch(query, tools)

    if not d.dispatched or not d.calls:
        # abstain: empty update, normal agent node handles the request
        return Command(goto="agent")

    msg = AIMessage(
        content="",
        tool_calls=[{"name": c["name"], "args": c["arguments"], "id": str(uuid.uuid4())}
                    for c in d.calls],
        response_metadata={
            "router": "thimble-fastpath",
            "vlp": d.vlp,
            "margin": d.margin,
            "vlp_threshold": _dispatcher.vlp_threshold,
            "margin_threshold": _dispatcher.margin_threshold,
            "latency_ms": d.ms,
        },
    )
    return Command(goto="tools", update={"messages": [msg]})
