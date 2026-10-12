# 第十二、十三步：论文 Agent 工具与引用

## 新增能力

- `runtime.py`：应用级 PaperRuntime，提供 catalog、检索、章节读取和导入 Worker；首次查询才加载 E5、ES 和 LlamaIndex 适配器。多 Session 可共享一个 Runtime。
- `plugin.py`：Provider 注册 paper_catalog/paper_ingestion/paper_retrieval，Consumer 注册 list_papers/search_papers/read_section 和 RAG 提示。
- `citations.py`：每个 Session 每轮独立 EvidenceLedger，签发随机 128 位 evidence_id，按 DeepSeek tokenizer 将完整证据 JSON 限制到 6000 tokens。相同证据及同位置完全被覆盖的段落去重；超预算整块跳过，不切坏公式或表格。
- `scripts/paper_chat.py`：沿用原 DeepSeekClient、Agent、Session、JsonlStore 运行论文对话，可 --resume。

## 运行

已配置本地 ES、模型和 DeepSeek API Key 后：

```powershell
.\.venv\Scripts\python.exe scripts/paper_chat.py "请解释 SparseBEV 的三个模块"
```

使用第十一步的独立验收库：

```powershell
.\.venv\Scripts\python.exe scripts/paper_chat.py "请解释 SparseBEV 的三个模块" --catalog .mini-harness/rag/ingestion-smoke.sqlite3 --index mini-harness-papers-ingestion-step11
```

需要的可选依赖仍是 rag-adapter、rag-embedding、rag-parser。只列论文或补读章节不需要模型加载；search 才初始化完整运行服务。问答会将选中证据发送到配置的 DeepSeek API。

## 实际调用关系

Agent → list_papers（提供精确章节路径）→ search_papers → PaperRuntime → LlamaIndexSearchAdapter → HybridSearchService → E5/ES/RRF → EvidenceLedger → 模型回答 → ID/版本/页码校验 → Session。

read_section 按 active_version 和章节前缀读取 catalog 中 SourceBlock。默认只返回作者正文；用户批注须显式 source_kind=user_annotation，来源列表由代码标明，不混为作者结论。

最终使用 `[cite:<本轮随机ID>]`。代码检查 ID 属于当前 Session 当前轮、版本仍有效、页码/位置存在；非法引用最多修复一次，仍失败则返回依据不足。被引用的原文及来源保存在 Session 的 rag/citations 事件中；恢复后新一轮需要重新检索签发 ID。

ID 的有效性不证明语义支持；最终来源列表是代码生成，但自然语言每一句是否被原文支持仍需要评审。对典型的“论文未记录/全文未提及”等由局部检索推出全文缺失的句式加了拦截，它不是完备的语义判断器。

## 上下文和生命周期

当前轮证据独立于会话表层保存在 Ledger，在每次 llm/request 前补入有预算的证据快照，防止原工具输出被 Spill/Prune 或摘要替换移走。下一轮清空映射，跨 Session 不共享引用 ID。

已测试真实 Spill、Prune 和 Session surface_op 摘要替换后仍保留当前轮证据。本项目尚未提供独立的 LLM Compact 执行器，因此这里验证的是压缩后的会话投影接口，不声称已经新增完整自动摘要器。

BundleConfig(enable_rag=True) 默认使用 vector-catalog.sqlite3；关闭时工具和提示不安装。若 Bundle 自行创建 Runtime，Context.dispose 负责关闭 ES/Worker；若调用方注入共享 rag_runtime，各 Session 只卸载自己的工具和 Ledger，由应用最终调用 runtime.close。Worker 默认不自启动，应用可调用 start_worker；停止后等待当前工作完成再关 ES。当前导入与在线查询共享串行锁，长导入会让查询等待，后续再做调度优化。

## 验证和局限

新增测试覆盖工具卸载、禁用开关、批注隔离、空范围拒绝、伪造/过期/跨 Session 引用、删除后引用失效、证据预算及上下文策略。

用户授权后进行了四类真实 DeepSeek 开发集验收：方法、3.2 节上下文、FreeText 批注、未提供的信息。完整答案和选中的原文在本地 `.mini-harness/rag/answer-eval/report.json`；包含个人批注，默认不纳入 Git。

逐条对照发现：方法与上下文主要结论有原文支持，批注标识与页码正确，无答案不应编造设备序列号。图表章节归属偶有偏差（如图2被归到上一章节），页码仍可定位；普通公式文本仍可能丢失排版；回答有时过长。不得把这四题的小样本结果当成普遍准确率，最终人工忠实度验收仍由使用者审阅。

第十四步 UI 上传、来源卡片和受限 PDF 页接口不在本次范围内。

最终回归：143 项，142 通过、1 跳过。无个人批注的验收摘要见 paper-agent-eval-summary.json。
