"""Deterministic evaluation helpers. No model calls or retrieval reruns."""
from hashlib import sha256
import json
from pathlib import Path
from .models import ArtifactReference, MetricResult
from .registry import METRICS, validate_metric_result


def result(metric_id, target, value=None, *, status="scored", reason=None, evidence_ids=(), numerator=None, denominator=None):
    definition = METRICS[metric_id]
    threshold = definition.threshold if status == "scored" else None
    passed = None if threshold is None else (value <= threshold if definition.direction == "lower" else value >= threshold)
    return validate_metric_result(MetricResult(metric_id=metric_id, scope=definition.scope,
        unit=definition.unit, target_id=str(target), status=status,
        value=float(value) if value is not None else None, reason=reason,
        threshold=threshold, passed=passed, evidence_ids=list(evidence_ids),
        numerator=numerator, denominator=denominator))


def read_artifact(root, ref):
    ref = ArtifactReference.model_validate(dict(ref))
    root = Path(root).resolve()
    path = (root / ref.relative_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Artifact outside attempt directory")
    if ref.size_bytes > 32 * 1024 * 1024:
        raise ValueError("Artifact exceeds evaluation read limit")
    raw = path.read_bytes()
    if len(raw) != ref.size_bytes or sha256(raw).hexdigest() != ref.sha256:
        raise ValueError("Artifact integrity mismatch")
    return json.loads(raw)


def retrieval_metrics(target, ranked, gold, *, top_k):
    names = [f"{name}_at_{k}" for k in (1, 3, 5) for name in ("recall", "hit", "mrr")]
    if not gold:
        return [result(name, target, status="missing_data", reason="No exact chunk-level gold for this query scope") for name in names]
    from ..rag.adapters.evaluation import evaluate_rankings
    scores = evaluate_rankings([(ranked, gold)])
    return [result(name, target, scores[name.replace("_at_", "@")]) if int(name[-1]) <= top_k
            else result(name, target, status="missing_data", reason="Requested top_k is smaller than metric K") for name in names]
