"""Metric catalog, not metric implementations. Shared contract for future UI."""
from dataclasses import asdict, dataclass
from types import MappingProxyType
from .models import MetricResult

@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    scope: str
    unit: str
    direction: str
    requires: tuple[str, ...]
    calculator: str
    description: str
    version: str = "1"
    threshold: float | None = None
    implementation: str = "planned"

def _catalog():
    rows = []
    def add(key, label, scope, unit, direction, requires, calculator, description,
            threshold=None, implementation="planned"):
        rows.append(MetricDefinition(key,label,scope,unit,direction,tuple(requires),
                    calculator,description,threshold=threshold,implementation=implementation))
    for key,label,req in (
        ("task_success","任务完成",("expected_behavior",)),
        ("requirement_coverage","需求覆盖",("key_points",)),
        ("answer_relevance","回答相关性",("input",)),
        ("answer_correctness","事实正确性",("reference_answer",)),
        ("faithfulness","证据忠实度",("injected_evidence",)),
        ("citation_support","引用支持率",("claim_citation_pairs",)),
        ("citation_coverage","引用覆盖率",("claims_requiring_citations",)),
        ("recovery_success","错误恢复",("expected_recovery",)),
    ):
        add(key,label,"case","ratio","higher",(*req,"actual_output"),"deepeval_or_human",
            "明确 rubric 后逐例评分；未标注或未评分时不可当成零")
    for key,label,req in (
        ("tool_correctness","工具调用正确性","expected_tools"),
        ("argument_correctness","工具参数正确性","tool_constraints"),
        ("precondition_satisfied","前置条件满足率","preconditions"),
        ("citation_validity","引用结构有效率","citation_locators"),
    ):
        add(key,label,"case","ratio","higher",(req,"trace"),"rules_or_deepeval","规则或明确参考标准判定")
    for key,label in (("required_refusal","应拒答命中率"),("false_refusal","可回答题误拒答率")):
        add(key,label,"run","ratio","lower" if key=="false_refusal" else "higher",
            ("answerable","refusal_labels"),"rules_or_human","仅在对应标注子集上计算，显示有效分母")
    for key,label in (("scope_violations","范围违规数"),("invalid_citations","无效引用数")):
        add(key,label,"case","count","lower",("trace","selected_scope"),"rules","硬性规则；零违规",threshold=0)
    for name,label in (("recall","证据召回率"),("hit","证据命中"),("mrr","首个相关结果倒数排名")):
        for k in (1,3,5):
            add(f"{name}_at_{k}",f"{label}@{k}","rag","ratio","higher",
                ("ranked_ids","gold_evidence"),"llamaindex","每次查询在前 K 个结果计算；跨调用覆盖另算",implementation="existing")
    for key,label in (("injected_gold_coverage","实际上下文证据覆盖率"),("case_gold_coverage","跨调用证据覆盖率")):
        add(key,label,"case","ratio","higher",("gold_evidence","evidence_flow"),"rules","相关证据集合去重后计算")
    for key,label in (("steps","Step 数"),("model_calls","模型请求数"),("tool_calls","工具调用数"),
                      ("retries","重试数"),("citation_repairs","引用修复数")):
        add(key,label,"turn","count","observe",("trace",),"telemetry","包括实际重试与修复调用",implementation="existing")
    for key,label in (("redundant_calls","重复无效调用率"),("budget_exceeded","预算超限率"),
                      ("execution_errors","执行错误率"),("judge_errors","评分错误率")):
        add(key,label,"run","ratio","lower",("trace","policy"),"rules","展示分母及失败分类")
    for scope in ("turn","step","model","tool","rag"):
        add(f"{scope}_elapsed_ms",f"{scope} 耗时",scope,"ms","lower",("timing",),"telemetry",
            "单调时钟墙钟时间；嵌套耗时不重复加总",implementation="existing")
    for key,label in (("p50_ms","耗时中位数"),("p95_ms","耗时 P95"),("queue_ms","排队耗时")):
        add(key,label,"run","ms","lower",("timing_samples",),"telemetry","按配置、冷热状态分组并显示样本量")
    add("throughput","吞吐","run","cases/s","higher",("completed_cases","wall_time"),"telemetry","完成数除以运行墙钟时间")
    for prefix in ("agent","judge"):
        for direction in ("input","output","cached_input"):
            add(f"{prefix}_{direction}_tokens",f"{prefix} {direction} token","run","tokens","observe",
                ("provider_usage",),"telemetry","来自 API；缺失不估填；缓存 token 是输入子集")
        add(f"{prefix}_cost",f"{prefix} 费用","run","currency","observe",("provider_usage","price_version","currency"),
            "pricing","Agent 与 judge 分开；缺失价格或 usage 显示不可计算")
    add("peak_rss_bytes","峰值内存","run","bytes","lower",("process_samples",),"telemetry","Python 与 ES 分开采样")
    for key,label in (("returned_evidence","返回证据数"),("accepted_evidence","接受证据数"),
                      ("injected_evidence","注入证据数"),("cited_evidence","引用证据数"),
                      ("deduplicated_evidence","去重证据数"),("budget_omitted_evidence","预算排除证据数")):
        add(key,label,"step","count","observe",("evidence_flow",),"telemetry","各阶段记录，不把去重与预算排除混合")
    for key,label in (("evidence_budget_ratio","证据预算占用率"),("deduplication_ratio","去重率"),("budget_omission_ratio","预算排除率")):
        add(key,label,"step","ratio","observe",("evidence_flow","budget"),"telemetry","明确分母，分母为零时不适用")
    for key,label in (("bm25_score","BM25 分数"),("vector_score","向量分数"),("rrf_score","RRF 分数"),("rerank_score","重排分数")):
        add(key,label,"rag","score","observe",("candidate_scores",),"retriever","不是概率；未启用该通道时不适用")
    add("tool_constraints_satisfied", "工具名称约束满足", "case", "ratio", "higher", ("trace", "tool_constraints"), "rules", "必需工具均调用且禁用工具未调用；不判断语义", threshold=1, implementation="implemented")
    add("tool_recoveries", "工具错误后成功次数", "case", "count", "observe", ("trace",), "rules", "同轮同名工具由失败转为成功；不等于任务恢复", implementation="implemented")
    from dataclasses import replace
    implemented = {"scope_violations", "invalid_citations", "citation_validity", "case_gold_coverage", "injected_gold_coverage"}
    rows = [replace(row, implementation="implemented") if row.metric_id in implemented else row for row in rows]
    return MappingProxyType({r.metric_id:r for r in rows})

METRICS = _catalog()

def metric_catalog():
    return [asdict(m) for m in METRICS.values()]

def validate_metric_result(result: MetricResult):
    definition = METRICS[result.metric_id]
    if (result.scope,result.unit,result.metric_version) != (definition.scope,definition.unit,definition.version):
        raise ValueError("Metric scope/unit/version does not match registry")
    if result.value is not None:
        if definition.unit in ("ratio",) and not 0 <= result.value <= 1:
            raise ValueError("Ratio must be between zero and one")
        if definition.unit in ("count","tokens","bytes","ms") and result.value < 0:
            raise ValueError("Measurement cannot be negative")
        if definition.unit in ("count","tokens","bytes") and not result.value.is_integer():
            raise ValueError("Count must be an integer")
    return result
