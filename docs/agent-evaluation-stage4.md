# 阶段 4：规则与检索评分

## 已完成

每个 attempt 的真实 Agent 执行结束后，runner 调用 rules.score_case，计算结果写入 SQLite metric_results 表以及本次报告。CLI status 会展示各 attempt 的指标。

- metrics.py：统一 MetricResult、按注册表校验单位/范围、校验产物路径/长度/SHA256；调用原有 rag/adapters/evaluation.py 的 LlamaIndex RetrieverEvaluator。
- rules.py：按 turn 重放调用、证据返回、上下文注入与最终引用，生成规则与检索结果。
- store.py：新增 metric_results，按 attempt、指标、目标、版本保存。保留旧 attempt，不混入后续重跑。
- runner.py：自动评分；评分错误单独记录，不能触发重新调用模型。
- registry.py：标记已实现规则，新增工具约束满足与工具恢复次数。

## 指标口径

| 指标 | 计算方式 |
|---|---|
| scope_violations | 请求超出该 turn 选中论文、read_section 版本不匹配、返回证据超出选中论文；被工具拦截的越界尝试也计入，表示 Agent 行为违规，不等于数据已经泄漏 |
| invalid_citations | 最终回答中不属于本轮接受证据且未通过引用事件确认的 ID，以及格式错误；修复前的草稿仍保留原轨迹，但不计入最终引用违规 |
| citation_validity | 最终回答有效引用 ID 数/引用 ID 总数；同轮重复 ID 去重；不评估引用是否语义支持结论 |
| tool_constraints_satisfied | required_tools 都被调用，forbidden_tools 都未被调用；未配置约束则不适用；不评价参数/顺序 |
| tool_recoveries | 同一 turn 的同名工具失败后再次成功的次数；连续同名失败合成一个恢复事件，不等于用户任务已经完成 |
| recovery_success | 有工具错误时保留为 missing_data，等待语义 rubric/judge；无错误时 not_applicable |
| Recall/Hit/MRR @1/3/5 | 重放每次查询的 returned_chunk_ids；金标取用例中实际请求论文范围内的 chunk IDs；完全复用历史 LlamaIndex evaluator |
| case_gold_coverage | 全用例返回过的唯一金标 chunk 数/金标 chunk 总数 |
| injected_gold_coverage | 有 rag/context 记录、且可追溯到本轮已接受证据的唯一金标 chunk 数/金标 chunk 总数 |
| steps/model_calls/tool_calls/retries/citation_repairs | 按 turn 的实际事件计数，重试和引用修复请求包含在 model_calls 内 |

范围和无效引用默认零违规；工具名称约束必须全部满足。任一硬规则失败会使质量 verdict=failed；规则全部通过仍为 incomplete，不能替代语义答案正确性。执行 status 与质量 verdict 分开。

## 缺失与比较约束

- 没有 exact chunk 金标、只有页码或 block 金标，不猜测 chunk 对应关系，检索/覆盖率标记 missing_data。
- 金标是 case 级相关集合，再限制到本次查询请求范围；不是每个改写 query 的人工独立标签，横向比较时应采用同一标签口径。
- top_k 小于 K 时不报告该 K 指标；top_k 足够但返回为空且有金标时记 0。
- 章节补读使用的 block 证据与 chunk 金标不能自动视为等价；当前覆盖率严格以 chunk ID 对齐，不代表语义覆盖率。
- 产物缺失/损坏与完整的空结果分开；读取时验证根路径和摘要。执行未完成时，不据不完整轨迹宣称零违规。
- 不会再次调用模型、ES 或 embedding。DeepEval 语义评估仍属于阶段 5；UI 展示属于阶段 6/7。
- metric_results 与报告文件分别提交，若报告发布失败，已经写入的指标仍能通过 attempt 查询。
- run 的 failed verdict 会保留已发生的硬规则失败；需要全新比较实验时创建新 run。

## 验证

评测相关测试覆盖：确定排名与历史指标公式一致、无金标/空结果/TopK不足、跨范围和伪造引用、返回与注入覆盖率、损坏产物、工具恢复与语义恢复区分、运行器隔离与恢复。测试使用临时文件与本地假模型，不调用真实模型 API。
