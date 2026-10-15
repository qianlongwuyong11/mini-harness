"""Request-only RAG projection; never rewrites the session journal."""
from dataclasses import replace
import json

RAG_TOOLS = {'list_papers', 'search_papers', 'read_section'}


def project_results(messages, current_results):
    # Match results to preceding calls, including providers that reuse IDs.
    owners, result_names, last_results = {}, {}, {}
    for index, message in enumerate(messages):
        for call in message.tool_calls:
            owners[call.id] = call.name
        if message.role == 'tool':
            result_names[index] = owners.get(message.tool_call_id)
            last_results[message.tool_call_id] = index
    projected = []
    for index, message in enumerate(messages):
        name = result_names.get(index)
        if name not in RAG_TOOLS:
            projected.append(message)
            continue
        current = (message.tool_call_id in current_results
                   and last_results[message.tool_call_id] == index)
        if current:
            summary = current_results[message.tool_call_id]
            # Current directory and errors are needed for planning/recovery.
            if name == 'list_papers' or summary is None:
                projected.append(message)
                continue
        else:
            summary = dict(status='historical', tool=name, evidence_ids=[],
                notice='历史工具记录，非本轮证据；旧目录不代表当前选择。需要原文时在当前范围重新检索或补读。')
        projected.append(replace(message, content=json.dumps(summary, ensure_ascii=False)))
    return projected
