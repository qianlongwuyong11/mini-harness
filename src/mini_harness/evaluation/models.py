"""Strict, framework-independent JSON contracts for evaluation workers and UI.

Requires the eval-contracts extra; importing these types never imports DeepEval.
Golden fields belong to the evaluator, never the Agent request.
"""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[int, Field(ge=0)]
Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

class PaperBinding(Contract):
    logical_key: Identifier
    paper_id: Text
    version: Text
    content_sha256: Digest

class GoldenEvidence(Contract):
    paper_key: Identifier
    content_sha256: Digest
    block_ids: list[Text] = Field(default_factory=list)
    chunk_ids: list[Text] = Field(default_factory=list)
    pages: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)

class TurnInput(Contract):
    input: Text
    # [] explicitly means no selected papers, never all papers.
    selected_papers: list[Identifier]

class Budgets(Contract):
    max_steps_per_turn: Annotated[int, Field(ge=1, le=100)] = 10
    timeout_seconds: Annotated[int, Field(ge=1)] = 120
    max_total_tokens: Annotated[int, Field(ge=1)] | None = None

class ToolConstraints(Contract):
    respect_selected_scope: bool = True
    required_tools: list[Identifier] = Field(default_factory=list)
    forbidden_tools: list[Identifier] = Field(default_factory=list)
    @model_validator(mode="after")
    def consistent(self):
        if set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError("A tool cannot be both required and forbidden")
        return self

class EvaluationCase(Contract):
    case_id: Identifier
    category: Identifier
    split: Literal["development", "regression", "heldout"]
    scenario: Text
    turns: list[TurnInput] = Field(min_length=1)
    expected_behavior: Text
    answerable: bool | None = None
    reference_answer: Text | None = None
    key_points: list[Text] = Field(default_factory=list)
    gold_evidence: list[GoldenEvidence] = Field(default_factory=list)
    tool_constraints: ToolConstraints = Field(default_factory=ToolConstraints)
    budgets: Budgets = Field(default_factory=Budgets)

class EvaluationDataset(Contract):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    papers: list[PaperBinding] = Field(default_factory=list)
    cases: list[EvaluationCase] = Field(min_length=1)
    @model_validator(mode="after")
    def references(self):
        ids = [c.case_id for c in self.cases]
        keys = [p.logical_key for p in self.papers]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("Duplicate case_id or paper logical_key")
        for case in self.cases:
            used = {p for t in case.turns for p in t.selected_papers}
            used.update(e.paper_key for e in case.gold_evidence)
            if not used <= set(keys):
                raise ValueError(f"Unbound paper in case {case.case_id}")
        return self

class TraceIdentity(Contract):
    run_id: Identifier
    case_id: Identifier
    attempt_id: Identifier
    session_id: Identifier | None = None
    turn_event_id: Count | None = None
    step_event_id: Count | None = None
    model_request_id: Identifier | None = None
    tool_call_id: Text | None = None
    span_id: Identifier | None = None
    parent_span_id: Identifier | None = None
    @model_validator(mode="after")
    def hierarchy(self):
        if self.step_event_id is not None and self.turn_event_id is None:
            raise ValueError("step requires turn_event_id")
        if (self.model_request_id or self.tool_call_id) and self.step_event_id is None:
            raise ValueError("model/tool requires step_event_id")
        if self.turn_event_id is not None and self.session_id is None:
            raise ValueError("turn requires session_id")
        if self.parent_span_id and (not self.span_id or self.span_id == self.parent_span_id):
            raise ValueError("Invalid parent span")
        return self

class ArtifactReference(Contract):
    relative_path: Text
    sha256: Digest
    size_bytes: Count
    media_type: Text = "application/json"
    redacted: bool = True
    @model_validator(mode="after")
    def safe_path(self):
        parts = self.relative_path.replace("\\", "/").split("/")
        if any(x in ("", ".", "..") for x in parts) or ":" in self.relative_path:
            raise ValueError("Artifact path must be relative without traversal")
        return self

class MetricResult(Contract):
    schema_version: Literal[1] = 1
    metric_id: Identifier
    metric_version: Text = "1"
    scope: Literal["run", "case", "turn", "step", "model", "tool", "rag"]
    target_id: Text
    status: Literal["scored", "not_applicable", "missing_data", "error"]
    value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    unit: Text
    threshold: Annotated[float, Field(allow_inf_nan=False)] | None = None
    passed: bool | None = None
    numerator: Count | None = None
    denominator: Annotated[int, Field(gt=0)] | None = None
    reason: Text | None = None
    evidence_ids: list[Text] = Field(default_factory=list)
    @model_validator(mode="after")
    def score_state(self):
        if self.status == "scored" and self.value is None:
            raise ValueError("A scored result needs a value")
        if self.status != "scored" and (self.value is not None or self.passed is not None):
            raise ValueError("Unavailable/error metrics cannot be zero or passed")
        if self.status != "scored" and not self.reason:
            raise ValueError("Unavailable metrics need a reason")
        if self.passed is not None and self.threshold is None:
            raise ValueError("A verdict requires an explicit threshold")
        if self.numerator is not None and (self.denominator is None or self.numerator > self.denominator):
            raise ValueError("Invalid ratio counts")
        return self

class RunRecord(Contract):
    schema_version: Literal[1] = 1
    run_id: Identifier
    dataset_sha256: Digest
    mode: Literal["live_rules", "live_judge", "rescore"]
    status: Literal["queued", "running", "cancelling", "completed", "failed", "interrupted"] = "queued"
    verdict: Literal["passed", "failed", "incomplete"] = "incomplete"
    configuration_sha256: Digest
    judge_model: Text | None = None
    @model_validator(mode="after")
    def judge_required(self):
        if self.mode == "live_judge" and not self.judge_model:
            raise ValueError("live_judge requires an explicit judge model")
        if self.status != "completed" and self.verdict == "passed":
            raise ValueError("An unfinished run cannot pass")
        return self
