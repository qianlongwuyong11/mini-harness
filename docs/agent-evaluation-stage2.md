# 阶段 2：RAG 明细与证据流

2026-10-13 完成。基于阶段 1 的内容寻址快照，补齐 RAG 运行事实；尚未新增评测页面或语义评分。

## 新事件

- rag/retrieval：LlamaIndex 检索 trace_id、状态及明细引用。文件包含原 query、关键词扩展 query、实际版本和来源过滤、两路候选正文/ID/分数/排名、RRF 各路贡献和最终顺序、返回 chunk ID、top_k 和融合配置。原始分数不是正确概率；当前没有重排序模型。
- rag/section：按论文版本、章节路径、来源类型读取的 SourceBlock 对应证据、页码/坐标、正文、数量、读取和打包总耗时。相似度分数不适用。范围校验之前失败时由 metrics/tool 保留错误码，不伪造成功读取记录。
- rag/evidence-flow：工具返回内容及每条证据的 accepted/duplicate/covered/budget/invalid_evidence 决定；接受时记录签发的 evidence_id。无效证据发生后仍维持原抛错行为，尚未处理的后续条目不计作已检查。
- rag/context：在实际模型请求边界记录 model_request_id 和证据 ID，以及最终 client 参数快照引用。每次引用修复与重试独立记录。空证据时无此事件。
- rag/citations：最终引用事件增加模型请求与 turn/step/session 身份，可以从最终引用回查账本和原工具调用。

工具内部 record_detail 继承 metric_scope 中的身份和 TraceStore；大内容保存为脱敏 JSON，JSONL 仅存引用、计数和状态。未处于 Agent scope 或未配置文件存储时不会擅自新增全局存储；有 scope 但没有 store 时标记 unavailable，写失败时标记 error。standalone 检索仍可直接取得 search_with_trace 的原结果。

## 关联路径

search_papers/read_section call_id → 返回证据 chunk_id → 账本 accepted 的 evidence_id → rag/context model_request_id → rag/citations。

检索候选与章节块 ID 可以不同：搜索基于 chunk，章节读取基于 SourceBlock。引用来源同时保留 paper/version/section/page，不以两个 ID 相同作为关联前提。

模型请求输入快照是最终上下文核验依据；rag/context 描述标准 RAG 插件注入的证据清单。未来若有其他插件在其后改写上下文，应对照最终 wire 快照，而不能只相信账本数量。当前快照仍记录脱敏内容，省略 reasoning_content。

## 展示与指标含义

step_metrics 新增 rag_details，保留检索、章节、证据筛选和上下文记录，供后续 UI 下钻。现有 UI 仍展示原有数值；阶段 6/7 才新增候选原文与证据流页面。

读了多少、接受多少、注入多少、引用多少是运行事实，不等于 Recall 或答案正确率。没有 gold 时不生成 Recall；没有 judge 时不生成证据支持度。RRF、BM25 和 ES 向量 _score 明确标注检索分数。

检索中途异常时原 hybrid 函数不返回完整 trace，rag/retrieval 标记 error；已完成阶段的 Callback span 可查看，但不声称失败请求也拥有完整候选快照。

## 验证

既有全量回归 175 项：174 通过、1 跳过。随后新增两项阶段 2 验证，RAG Agent 10 项和 telemetry 7 项通过：

1. 同一批证据的接受、重复和超预算原因可分别识别。
2. 真实 Agent loop（模拟模型）执行 read_section，原文章节进入账本、注入模型、最终引用的 ID 完整对应，step 聚合保留明细引用。

未调用远程模型；检索排序和引用判定规则没有因本次记录功能改变。运行中服务需重启才使用新增埋点。
