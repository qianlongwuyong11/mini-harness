# 第十一步：持久化 PDF 导入闭环

入口：`scripts/ingest_pdf.py`；实现：`src/mini_harness/rag/ingestion.py`。

## 流程与存储

PDF → 校验并复制原文件 → SQLite 入队 → PARSING → CHUNKING → EMBEDDING → INDEXING → PUBLISHING → READY。

- 原文件先用临时文件、flush/fsync、原子改名保存到 `<root>/sources/<sha256>.pdf`，再创建 SQLite 任务。原始文件夹的文件随后移走不影响任务。
- Docling 普通解析，不启用公式增强。产物保存在 `<root>/parsed/<identity>/`，包括 source.pdf、docling.json、document.md、source-blocks.json、parse-report.json。
- `catalog.save_blocks()` 将每块完整 JSON 写入 `<catalog路径>.blocks/<digest>.json`，再提交 SQLite 引用，随后进入 CHUNKING。
- 结构分块后将正文 chunks 保存到 `<root>/artifacts/<job摘要>/chunks.json`，再进入 EMBEDDING。注释块仍保存在解析产物与 catalog，但本期只索引论文正文。
- 向量按批生成，进度写入 SQLite；全部完成后原子写入 `<chunks内容+模型配置摘要>.vectors.json`，再进入 INDEXING。文件中 chunk_id 对应向量，不在 SQLite 塞入大向量。
- ES Bulk 逐项错误检查沿用既有实现；refresh 后检查数量、ID、完整块内容和向量可搜索性，最后 SQLite 单事务完成版本 READY、active_version 切换和任务完成。

## 运行

在项目根目录运行，需已经配置本地 ES、Docling 和选定 E5 模型：

```powershell
# 直接导入并等待完成
.\.venv\Scripts\python.exe scripts/ingest_pdf.py import "C:\path\paper.pdf"

# 只入队，再在另一个终端长期运行 Worker
.\.venv\Scripts\python.exe scripts/ingest_pdf.py enqueue "C:\path\paper.pdf"
.\.venv\Scripts\python.exe scripts/ingest_pdf.py work

# 查看、显式重试、取消；重试后由 Worker 消费
.\.venv\Scripts\python.exe scripts/ingest_pdf.py status "job-ID"
.\.venv\Scripts\python.exe scripts/ingest_pdf.py retry "job-ID"
.\.venv\Scripts\python.exe scripts/ingest_pdf.py cancel "job-ID"
```

默认 catalog 为 `.mini-harness/rag/vector-catalog.sqlite3`，root 为 `.mini-harness/rag/ingestion`。可用 `--catalog`、`--root`、`--index` 指定独立环境；重启时保持配置一致。`work --once` 只消费一项。

## 恢复、取消与并发

- 同一 catalog 的 Worker 通过 OS 文件锁排他运行，每次只有一篇在处理；另开 Worker 无法并行抢占处理资源。这是本机模式，不声称支持分布式部署。
- 心跳线程默认每 100 秒续租一次，租期 300 秒；它与阶段更新通过进程内锁串行使用最新任务快照。服务可由受管理线程调用 `serve(stop_event)`，设置 stop 后当前任务处理完再退出。
- 崩溃后等待租约到期，再由 Worker 恢复为排队状态。失败任务保留失败阶段和错误，需显式 retry；不无限重试永久错误。
- 重试从 PARSING 状态重新走流程，复用 catalog 中已校验的解析块；分块重新确定性生成；完整向量缓存可复用，半途向量化会重新编码。ES 写入使用稳定 ID 覆盖，成功前不影响旧 active_version。
- 取消在 SQLite 原子撤销任务的所有权。正在执行的 Docling/模型/ES 调用不会被强杀，返回后的检查阻止继续推进或发布；已经 READY 的任务不能按取消撤销。
- 中间产物和不可查询的暂存 ES 文档保留以便追溯，本步不自动删除原文件或实现垃圾回收。UI 生命周期接入属于后续步骤。

## 验证

8 项新增故障测试：重复导入与源文件移除后处理、失败保留旧版本及重试缓存复用、解析/编码失败阶段、写入中取消、发布前中断恢复、Worker 锁互斥、refresh 校验失败和长任务续租。

真实 SparseBEV 验证使用独立的 `.mini-harness/rag/ingestion-smoke.sqlite3`、`.mini-harness/rag/ingestion-smoke/`、ES 索引 `mini-harness-papers-ingestion-step11`。结果见 ingestion-report.json。

2026-10-12 实测：216 个 SourceBlock 和 216 个块 JSON；106 个正文 chunk 发布 READY；全新进程返回 5 条中文查询证据；重复导入 attempts 仍为 1。完整回归 136 项，135 通过、1 跳过。
