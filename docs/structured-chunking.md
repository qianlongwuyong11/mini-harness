# 第六步：结构化分块与引用定位

2026-10-12：已实现并完成 SparseBEV 样本验证。入口为 `rag.chunking.StructuredChunker.split(version, blocks)`，符合现有 Chunker 协议。

## 分块规则

- 以 SourceBlock 段落为单位，同章节、同 source_kind 的连续正文和公式在预算内合并。只拆分超限正文，不按字符数估算 token。
- 标题、图注、批注分别独立成块。公式保持完整，超限时明确报错，不截断数学表达。
- 表格独立处理；超限时保留表注并重复完整表头，按连续完整行分组。单行加表头仍超限时报错，交由后续大表处理策略解决。
- 超长文字用 tokenizer 的字符偏移定位切点，再对原始字符串切片；保留 Unicode，不将 token 解码结果当作原文。每个片段重新计数。
- 每块保留 section_path（完整父章节路径）、source_block_ids、所有来源页码与 bbox、前后 chunk 引用。source_kind 不混合，相邻引用也不会从作者正文跨到用户批注。
- 块类型、原始父块引用、解析警告可通过 source_block_ids 追溯，验收产物额外提供 chunk-metadata.json。parent_chunk_id 暂为空：当前没有虚构的父级摘要 chunk，父章节使用 section_path 表达。
- bbox 精度为原始 SourceBlock 范围；一个段落拆成多块时复用原段落坐标，不声称得到逐 token 坐标。

## Tokenizer 与版本

第七步尚未选定 Embedding 模型，本轮仅使用候选 `intfloat/multilingual-e5-small` 的真实 tokenizer，不加载权重。最大输入预算 384 tokens，计数包含 `passage: ` 前缀和特殊 token，关闭自动截断和 padding。后续查询前缀和 embedding 输入应由对应模型适配器统一管理。

tokenizer 文件 SHA-256、模型标识、前缀、预算和算法版本共同决定 chunker_version；chunk_id 由 paper_id、文档版本、chunker_version 和序号生成。配置改变必须新建版本。第七步换模型时使用新 tokenizer 重新分块，不能复用本次计数作为新模型的长度。

## 验收结果

- 输入：第五步普通解析的 216 个块（195 个论文块、21 个批注块）。未启用公式增强。
- 输出：127 个 chunk，最大 382 tokens，全部输入块均有引用，重复运行内容和 ID 一致。
- 检查 3.2 节：公式（2）、（3）及相关解释在同一块；公式（1）后的说明因原阅读顺序中的图注而分离，但前后引用保留。后续检索阶段须根据这些引用扩展邻近上下文，不能假定单块永远包含全部解释。
- 长表按整行拆分；保留完整标题、表头和原始表格定位。不会修复上游表格或公式的识别错误。
- 生成 chunks.json、source-blocks.json、chunk-metadata.json 和可读 chunks.md；采用临时文件、fsync、原子替换。它们是本地验收产物，不会将 SQLite 中版本发布为 READY，也不写入 ES。正式后台任务的阶段衔接沿用后续工作流计划。
- 回归测试：110 项，109 通过、1 跳过。

## 复现

在项目目录运行（需第五步解析产物及下载好的 tokenizer）：

```powershell
.\.venv\Scripts\python.exe scripts/chunking_eval.py .mini-harness/rag/parse-eval/1f75ba27b113731af6a73f52/source-blocks.json
```

汇总位于 `docs/sparsebev-chunking-report.json`，产物路径在其 artifacts 字段。命令不调用 PDF 模型或 Embedding 模型，不联网。tokenizer 尚未缓存时可先使用 huggingface_hub 下载该模型的 tokenizer.json，文件指纹记录在验收报告。
