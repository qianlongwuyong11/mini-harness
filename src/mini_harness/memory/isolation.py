"""Request-local isolation for memories without trusted paper provenance."""
from dataclasses import replace
import json


def paper_scope_key(ctx):
    if "paper_scope" not in ctx.service_names:
        return None
    ids = ctx.require("paper_scope")["paper_ids"]
    return None if ids is None else tuple(sorted(set(ids)))


def memory_allowed(ctx):
    # Generic memories cannot prove which selected paper they originated from.
    return paper_scope_key(ctx) is None


def redact_memory_results(messages):
    owners = {}
    result = []
    for message in messages:
        for call in message.tool_calls:
            owners[call.id] = call.name
        if message.role == "tool" and owners.get(message.tool_call_id) in {"memory_search", "memory_read"}:
            message = replace(message, content=json.dumps({
                "status": "scope_blocked", "notice": "Memory lacks trusted paper provenance; use current paper tools."}))
        result.append(message)
    return result
