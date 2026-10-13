"""Replay captured events into deterministic rules and LlamaIndex metrics."""
import json
from ..rag.citations import CITATION
from .metrics import result, read_artifact, retrieval_metrics


def score_case(case, dataset, events, artifact_root, *, target_id=None):
    target = target_id or case.case_id
    output = []
    bindings = {p.logical_key: p for p in dataset.papers}
    versions = {p.paper_id: p.version for p in dataset.papers}
    turns = []
    current = None
    for event in events:
        if event.type == "turn/start":
            current = []
            turns.append(current)
        if current is not None:
            current.append(event)
    scope_errors, invalid_citations, citations, tool_names = [], [], 0, set()
    unknown_scope = False
    complete = len(turns) == len(case.turns) and all(any(e.type == "turn/end" and e.data.get("reason") == "completed" for e in turn) for turn in turns)
    recovered, failures = 0, 0
    gold_all = {(bindings[g.paper_key].paper_id, bindings[g.paper_key].version, chunk)
                for g in case.gold_evidence for chunk in g.chunk_ids}
    exact_gold = bool(gold_all) and all(g.chunk_ids for g in case.gold_evidence)
    returned_gold, injected_gold = set(), set()
    flow_missing = False
    for index, turn in enumerate(turns):
        selected = {bindings[key].paper_id for key in case.turns[index].selected_papers} if index < len(case.turns) else set()
        accepted = {}
        pending_errors = set()
        call_scopes = {}
        successful_rag_calls = set()
        recorded_flows = set()
        recorded_retrievals = set()
        final_citation_ids = set()
        final_text = None
        per_turn = f"{target}:turn:{turn[0].id}"
        for kind, metric in (("step/start", "steps"), ("metrics/model", "model_calls"), ("tool/call", "tool_calls"), ("rag/repair", "citation_repairs")):
            output.append(result(metric, per_turn, sum(e.type == kind for e in turn)))
        output.append(result("retries", per_turn, sum(e.type == "metrics/model" and e.data.get("purpose") == "retry" for e in turn)))
        for event in turn:
            data = event.data
            if event.type == "tool/call":
                name = data.get("name")
                tool_names.add(name)
                if name in ("search_papers", "read_section"):
                    try:
                        args = json.loads(data["arguments"])
                        ids = args.get("paper_ids", []) if name == "search_papers" else [args.get("paper_id")]
                        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                            raise ValueError("Invalid paper scope")
                        call_scopes[data.get("call_id")] = set(ids)
                        if not set(ids) <= selected or (name == "read_section" and args.get("version") != versions.get(args.get("paper_id"))):
                            scope_errors.append(str(event.id))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        unknown_scope = True
            elif event.type == "metrics/tool":
                name = data.get("name")
                if data.get("status") == "error":
                    failures += 1
                    pending_errors.add(name)
                elif name in pending_errors:
                    recovered += 1
                    pending_errors.remove(name)
                if data.get("status") == "ok" and name in ("search_papers", "read_section"):
                    successful_rag_calls.add(data.get("call_id"))
            elif event.type == "rag/evidence-flow":
                recorded_flows.add(data.get("call_id"))
                try:
                    detail = read_artifact(artifact_root, data["detail_ref"])
                    for item in detail["returned"]:
                        key = (item["paper_id"], item["version"], item["chunk_id"])
                        returned_gold.add(key)
                        if item["paper_id"] not in selected:
                            scope_errors.append(str(event.id))
                    for decision in detail["decisions"]:
                        if decision.get("reason") == "accepted":
                            accepted[decision["evidence_id"]] = (decision["paper_id"], decision["version"], decision["chunk_id"])
                except (ValueError, TypeError, OSError, KeyError):
                    flow_missing = True
            elif event.type == "rag/context":
                for identity in data.get("evidence_ids", []):
                    if identity in accepted:
                        injected_gold.add(accepted[identity])
                    else:
                        flow_missing = True
            elif event.type == "rag/citations":
                final_citation_ids = set(data.get("ids", []))
            elif event.type == "assistant/message" and not data.get("tool_calls"):
                final_text = data.get("content") or ""
            elif event.type == "rag/retrieval":
                recorded_retrievals.add(data.get("call_id"))
                query_target = f"{target}:query:{event.id}"
                try:
                    if data.get("status") != "ok":
                        raise ValueError("Retrieval failed")
                    detail = read_artifact(artifact_root, data["detail_ref"])
                    ranked = detail["returned_chunk_ids"]
                    # Case-level labels restricted to selected papers, never page IDs guessed as chunks.
                    query_scope = call_scopes.get(data.get("call_id"), selected)
                    gold = [chunk for paper, version, chunk in gold_all if paper in query_scope] if exact_gold else []
                    output.extend(retrieval_metrics(query_target, ranked, gold, top_k=detail["top_k"]))
                except (ValueError, TypeError, OSError, KeyError, ImportError) as error:
                    for k in (1, 3, 5):
                        for name in ("recall", "hit", "mrr"):
                            output.append(result(f"{name}_at_{k}", query_target, status="error", reason=f"Retrieval evaluation unavailable: {type(error).__name__}"))
        if successful_rag_calls - recorded_flows:
            flow_missing = True
        for call_id in set(call_scopes) - recorded_retrievals:
            call = next((e for e in turn if e.type == "tool/call" and e.data.get("call_id") == call_id), None)
            if call is not None and call.data.get("name") == "search_papers":
                for k in (1, 3, 5):
                    for name in ("recall", "hit", "mrr"):
                        output.append(result(f"{name}_at_{k}", f"{target}:call:{call.id}", status="missing_data", reason="No successful captured retrieval ranking"))
        if final_text is not None:
            ids = set(CITATION.findall(final_text))
            citations += len(ids)
            invalid_citations.extend(sorted(ids - (final_citation_ids & set(accepted))))
            if "[cite:" in CITATION.sub("", final_text):
                invalid_citations.append(f"malformed:turn:{index}")
                citations += 1
    def count_or_missing(metric, count, evidence=()):
        return result(metric, target, count, evidence_ids=evidence) if (complete and not flow_missing) or (count > 0 and not flow_missing) else result(metric, target, status="missing_data", reason="Incomplete execution or evidence trace")
    output.append((count_or_missing("scope_violations", len(set(scope_errors)), sorted(set(scope_errors))) if not unknown_scope else result("scope_violations", target, status="missing_data", reason="Malformed scope arguments")) if case.tool_constraints.respect_selected_scope else result("scope_violations", target, status="not_applicable", reason="Scope rule disabled in dataset"))
    output.append(count_or_missing("invalid_citations", len(invalid_citations), invalid_citations))
    output.append(result("citation_validity", target, (citations-len(invalid_citations))/citations) if complete and not flow_missing and citations and len(invalid_citations) <= citations else result("citation_validity", target, status="not_applicable" if not citations else "missing_data", reason="No citations or incomplete citation trace"))
    required, forbidden = set(case.tool_constraints.required_tools), set(case.tool_constraints.forbidden_tools)
    if complete and (required or forbidden):
        output.append(result("tool_constraints_satisfied", target, float(required <= tool_names and not forbidden & tool_names), reason="Checks tool names only, not arguments or answer quality"))
    else:
        output.append(result("tool_constraints_satisfied", target, status="missing_data" if not complete else "not_applicable", reason="Incomplete execution or no tool constraints"))
    output.append(result("tool_recoveries", target, recovered, reason="Same tool name later succeeded in the same turn; not semantic recovery"))
    output.append(result("recovery_success", target, status="missing_data" if failures else "not_applicable", reason="Semantic recovery requires a rubric/judge" if failures else "No tool error observed"))
    for metric, seen in (("case_gold_coverage", returned_gold), ("injected_gold_coverage", injected_gold)):
        if exact_gold and complete and not flow_missing:
            count = len(gold_all & seen)
            output.append(result(metric, target, count/len(gold_all), numerator=count, denominator=len(gold_all)))
        else:
            output.append(result(metric, target, status="missing_data", reason="Requires complete trace and exact chunk-level gold for all labels"))
    return output
