# 检索评估与每轮运行统计

## 实现与统计口径

- `rag/adapters/evaluation.py` 使用 LlamaIndex `RetrieverEvaluator` 的 Recall、Hit Rate、MRR，在 K=1/3/5 评估固定标注集。捕获实际检索排名后交给评估器复算，不重复请求模型或 ES。空结果按零分处理；当前安装版本的评估器会拒绝空结果，因此适配层统一处理。
- `rag/adapters/tracing.py` 使用 LlamaIndex `CallbackManager` / `BaseCallbackHandler`。每次查询独立追踪，记录父子事件 ID、阶段耗时和成功/失败。BM25、query embedding、kNN、融合阶段显式接入回调；外层 retrieve 事件由 LlamaIndex Retriever 产生。
- `telemetry.py` 汇总模型、工具与 RAG 事件；`agent.py` 在真实调用边界计时，包含重试和引用修复引起的额外调用。统计随会话 JSONL 保存，恢复会话后可查看。
- UI 在每个 step 的输出后显示紧凑入口，点击打开单个执行详情面板。旧对话不补算缺失指标。当前统计主 Agent；子 Agent 独立会话尚未汇总。工具耗时包含内部 RAG 等阶段，不能把所有耗时直接相加。
- token 取模型 API 返回的 usage，不用字符估算替代账单。缺失显示“未提供”，部分缺失标记不完整；失败请求的实际费用可能未知。证据预算的本地估算 token 单独展示。
- LlamaIndex 追踪仅保存本地数值和技术元数据，不增加外部追踪服务，不复制回调中的问题、论文正文或密钥。UI 另从原有工具事件提取有长度限制的参数和结果预览，不把预览写回统计事件。

## 指标用途

| 指标 | 后续优化用途 |
|---|---|
| 每轮总耗时、模型调用次数/耗时/失败、输入输出 token | 判断等待来自模型还是工具，发现上下文膨胀和重复请求 |
| 缓存命中输入 token（服务提供时） | 观察前缀缓存收益 |
| 工具次数、逐次耗时和失败 | 定位反复调用与慢工具 |
| 重试次数、计划等待、同一步额外调用 | 区分正常推理与重试/修复成本；计划等待不等于实际睡眠时长 |
| RAG 初始化与锁等待 | 区分首次加载模型、检索本身和并发排队 |
| BM25 / query embedding / kNN / 融合耗时 | 定位查询编码、ES 或融合瓶颈 |
| 两路候选数、去重候选数、返回数 | 发现空召回、过滤过严和两路互补程度 |
| 证据新增/保留数、去重或预算排除数、证据预算 | 观察上下文是否被占满；排除数不区分去重与预算原因 |
| 最终引用数量、引用修复次数 | 发现引用格式与证据使用问题，不把数量当作正确率 |
| 离线 Recall / Hit / MRR | 用标准答案衡量覆盖率、命中率、排序质量 |

普通问题没有 gold，不计算在线 Recall/MRR。当前非流式模型调用没有首 token 延迟；没有计费价格，因此不估算金额。跨轮 P50/P95、子 Agent 汇总可在积累真实样本后追加。

## 验证

仓库根目录：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/rag_acceptance.py --replay-report docs/rag-acceptance-report.json --output docs/rag-llamaindex-evaluation.json
# 需本地 ES 运行、默认论文索引和本地 E5 模型
.\.venv\Scripts\python.exe scripts/rag_trace_verify.py
```

`rag-llamaindex-evaluation.json` 是同一批排名的官方评估复算，不代表重新检索或通过此前未通过的质量门槛。`rag-trace-verification.json` 是真实本地 ES + E5 查询记录，没有调用远程聊天模型。

更新 UI 后需要重启原 Python UI 服务并刷新页面，新一轮对话才有完整统计。

## Turn 与 Step

- `Agent.run` 从 inbox 取到一组用户输入后开始 turn。UI 中通常一次发送对应一个 turn；底层支持排队输入和 step 间的 steer 消息。
- `_run_turn` 每次 `step/start` 后准备请求、调用模型、写入回复，并串行执行这次回复中的所有工具，最后写 `step/end`。当前最多 10 个 step。
- 无工具调用且没有 steer 输入时，step 和 turn 随后结束；有工具调用时，工具结果留给下一 step 的模型请求。
- 同一 step 可能因重试或引用修复出现多个实际模型请求。模型/工具/RAG 事件通过 turn 起始事件 ID、step 起始事件 ID 和 call_id 关联，不依赖会在恢复后重置的计数器。
- 新 step 总耗时使用 perf_counter；`agent/pre-step` 插件执行发生在 step/start 之前，不计入这个时长。旧记录以事件时间差估算。总耗时包含模型、工具及请求准备，内部 RAG 耗时不可重复累加。
- `step_metrics` 根据原始事件重建统计，包含工具结果预览；新界面不再把所有 turn 汇总卡片插到聊天末尾。没有计量的旧记录不伪造 token 或模型耗时。
