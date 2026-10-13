"""Validated dataset snapshots. Only TurnInput is projected into the agent."""
from hashlib import sha256
import json
from pathlib import Path

from .models import EvaluationDataset

MAX_DATASET_BYTES = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_dataset(root, relative_path):
    root = Path(root).resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or relative.drive or ":" in str(relative) or ".." in relative.parts:
        raise ValueError("Dataset must be a relative path beneath the selected directory")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix.lower() not in (".json", ".jsonl"):
        raise ValueError("Dataset path/type outside selected scope")
    with path.open("rb") as handle:
        raw = handle.read(MAX_DATASET_BYTES + 1)
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError("Dataset exceeds 8 MiB")
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line, object_pairs_hook=_unique) for line in text.splitlines() if line.strip()]
        if not rows or not isinstance(rows[0], dict) or "cases" in rows[0]:
            raise ValueError("JSONL first line must be a manifest without cases")
        value = {**rows[0], "cases": rows[1:]}
    else:
        value = json.loads(text, object_pairs_hook=_unique)
    return EvaluationDataset.model_validate(value)


def bind_papers(dataset, catalog):
    """Require exact active version/hash; titles are never identifiers."""
    mapping = {}
    for binding in dataset.papers:
        paper = catalog.get_paper(binding.paper_id)
        version = catalog.get_active_version(binding.paper_id)
        if (paper is None or version is None or version.version != binding.version
                or version.source_hash != binding.content_sha256):
            raise ValueError(f"Paper version/hash mismatch: {binding.logical_key}")
        mapping[binding.logical_key] = binding.paper_id
    for case in dataset.cases:
        for evidence in case.gold_evidence:
            binding = next(p for p in dataset.papers if p.logical_key == evidence.paper_key)
            if evidence.content_sha256 != binding.content_sha256:
                raise ValueError(f"Gold evidence source mismatch: {case.case_id}")
    return mapping
