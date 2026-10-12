# 第八步：Elasticsearch 向量入库与查询

2026-10-12 已完成。物理索引为 `mini-harness-papers-dense-5a0b4797e75c38bd`，含106条 SparseBEV 作者论文 chunk。每条同时保存正文、章节、页码、完整 Chunk、模型指纹和768维向量；用户批注保留在来源文件中，未加入本次作者证据索引。

## 数据与接口

- `rag/vector_index.py`：ElasticsearchVectorIndex，在原有 BM25 适配器上增加向量字段与 kNN；BM25-only 类保持可用。
- dense_vector 使用 float、dot_product、显式 hnsw，m=16、ef_construction=100。输入必须为有限数、维度一致、L2归一化。显式指定 HNSW 避免随 ES 默认量化策略变化。
- 模型ID、commit revision、tokenizer SHA-256、维度、前缀、池化、归一化、dtype及输入上限共同形成索引指纹，写入 mapping `_meta`。模型变化创建新物理索引；对原索引使用不一致配置会报错。
- `VectorSearchService` 在构造时核对编码器与索引的完整配置，然后执行中文 query 编码与 kNN。低层 vector_search 接收裸向量，仅能检查维度/数值/归一化；业务应使用绑定编码器的服务，不能仅凭相同维度判断向量兼容。
- 过滤放在 `knn.filter` 中：active paper/version成对过滤、source_kind、模型指纹；返回后再查SQLite，剔除查询途中被删除或失效的版本。当前本地实现最多512个选中活跃版本。
- `rag/vector_pipeline.py` 串联分批编码、心跳、Bulk、refresh、计数与ID/正文核验，最后调用SQLite complete_job，在一次事务中发布READY、切换active_version和结束任务。Bulk HTTP 200 也逐项检查错误。
- ES 刷新后验证所有期望 chunk ID、正文、模型指纹和可搜索向量数量。失败任务记录错误，版本保持STAGING，旧active版本不变。重试使用相同chunk_id覆盖，不增加重复记录。
- 编码分批、ES请求按条数与字节数分批；目前单篇所有向量暂存在Python内存，后续海量文件流水线可改为持久批次。模型运行与ES调用的超时仍需在后续后台任务层统一治理。

参考：[Elastic dense_vector](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector)、[kNN 前置过滤](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query)。

## 持久化与发布

本阶段独立演示目录：`.mini-harness/rag/vector-catalog.sqlite3`，以及相邻 `.blocks` 原文块存储。SQLite注册原PDF路径并核对SHA-256，ES数据在之前安装的本地ES数据目录中。备份时需同时备份SQLite、原文块、原PDF与ES数据/快照。

本次发布版本：`557c3224b3e11dd5d1372e20`，模型 `intfloat/multilingual-e5-base`、revision `d128750597153bb5987e10b1c3493a34e5a4502a`。`selected/` 中原文件仍标记 staging，它是不可变入库输入；是否已发布以SQLite当前状态为准。

同一 ready 版本重复执行 import 只做完整性核验；不会覆盖已发布数据。已有 failed 任务可重试，尚有有效租约的任务不会被抢占。新增可选 claim_job(job_id=...) 防止导入指定论文时抢走其他论文任务。

## 实测结果

- 106条真实论文块重新批量编码、写入、核验、发布：约35.90秒，不含模型加载。
- 新进程+ES停止重启后，数量仍为106，12个问题的Top5排名逐项一致。
- 查询总耗时包括：query编码、SQLite active版本检查、ES HTTPS kNN请求与结果解码、返回后版本检查。
- 重启后36次查询中位54.93ms，P95 63.22ms；重启前中位60.22ms。排除模型启动、Agent编排、LLM生成及UI传输，因此不是最终聊天的端到端耗时。
- 本篇12题 Recall@5=70.83%，Hit@5=75%；仍是第七步的小样本标签，不能外推为普遍准确率。
- 真实ES故障/生命周期验收：幂等重试、staging隐藏、论文/批注过滤、HTTP200下部分Bulk失败、失败新版本不替换旧版、修复重试后切换、模型版本错配拒绝、软删除隐藏。
- 回归测试：120项，119通过、1跳过。详情见 `vector-index-report.json`、`vector-lifecycle-report.json`。

## 使用

在项目目录打开两个终端。第一个启动本地ES：

```powershell
.\.venv\Scripts\python.exe scripts/local_es.py serve
```

第二个导入已有的第五至七步产物，或查询已发布数据：

```powershell
.\.venv\Scripts\python.exe scripts/vector_search.py import
.\.venv\Scripts\python.exe scripts/vector_search.py query "自注意力如何调整每个查询的感受野？" --top-k 5
```

每次CLI查询都会新建进程并加载模型，实际命令等待时间包含模型加载，明显长于上面的常驻模型查询耗时。后续Agent服务启动时持有一个编码器实例即可避免逐次加载。

当前CLI默认演示SparseBEV的已有结构化产物；完整文件夹导入、后台任务、Agent工具和UI集成按后续步骤实现。第九步接入 BM25+向量结果的 RRF 融合。
