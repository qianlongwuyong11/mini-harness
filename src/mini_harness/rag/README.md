# RAG 插件边界

默认 enable_rag=False，不导入 RAG Provider，不创建论文目录数据库。
开启时注册 paper_catalog；当前不启动 Docling、ES、Embedding 或后台 Worker。

## 第二步：SQLite 目录与持久任务

实现：catalog.SQLitePaperCatalog。默认数据位置：

    .mini-harness/rag/catalog.sqlite3
    .mini-harness/rag/catalog.sqlite3.blocks/<sha256>.json

BundleConfig.rag_catalog_path 可以覆盖默认路径。SQLite 使用标准库 sqlite3，
每次操作打开并关闭连接，启用外键、WAL 和 FULL synchronous。写操作使用
BEGIN IMMEDIATE，不共用跨线程 Connection；10秒锁等待超时会抛错给调用方。
只面向本机磁盘，不把 SQLite 放在网络共享盘作为分布式数据库。

数据库包含 papers、versions、blocks、jobs、active_versions 五张表，
PRAGMA user_version=1 标识迁移版本；拒绝打开未知新版本。
blocks 表保存块ID、顺序及内容文件摘要引用，具体正文、页码、bbox、章节和
关系位于完整 SourceBlock JSON 中；SQLite 不存向量，也不重复保存整篇正文。
保留 JSON 是为了无损重建自有模型，不能把正文当作指令执行。

## 使用示例

    from mini_harness.rag.catalog import SQLitePaperCatalog

    catalog = SQLitePaperCatalog("data/rag/catalog.sqlite3")
    job = catalog.enqueue_import(
        source_uri="file:///F:/papers/immutable-copy.pdf",
        source_hash="<原始PDF字节的SHA-256>",
        parser_version="docling-config-v1",
        chunker_version="chunk-config-v1",
    )
    claimed = catalog.claim_job("worker-1")

source_uri 必须由将来的导入服务指向已经保存的不可变源文件；本步骤不会复制
PDF或校验源文件内容。解析与分块版本需包括所有影响输出的配置；Embedding
模型和revision也是版本身份的一部分。同一内容、同一配置的提交复用同一任务，
即使来自另一个文件路径；换配置则建立新版本。论文、版本、任务在同一事务创建。

create_paper/save_version/save_job 是低层创建接口；save_job 不允许覆盖已有状态。
正常业务使用 enqueue_import，FAILED 任务显式 retry_job，不通过再次提交自动重试。

## 状态与恢复

    queued/waiting
        -> running/parsing
        -> running/chunking
        -> running/embedding
        -> running/indexing
        -> running/publishing
        -> ready/completed

运行中的任务可 fail_job，保留失败阶段和错误；retry_job 从 parsing 重新处理。
每次领取增加 attempts。Worker 通过 heartbeat 续租；插件启动只回收已过期租约，
不会抢走另一个仍在运行的Worker的任务。之后由Worker定时调用 recover_expired_jobs。
状态更新要求传入上次操作返回的完整 IngestionJob 快照；旧快照、过期租约、
非法阶段跳转会抛 CatalogConflict。心跳和阶段推进需在Worker内协调，发生快照
冲突后重新读取再决定是否继续，不能无条件覆盖。

save_blocks 接收一个版本完整、有序的块列表，不是逐批追加。
相同快照重试幂等，不同内容必须使用新版本。先落文件再提交数据库引用，
崩溃可能留下无引用的文件；本期保留它们，不自动清理。读取校验SHA-256，
缺失或损坏会报错，不能发布为空索引。备份时应一并备份数据库和 .blocks 目录。

complete_job 要求处于 publishing、存在可读原文块、索引数量一致且可检索，
随后在一个SQLite事务内完成：版本READY、active_version切换、任务READY。
第八步已接入 vector_pipeline：实查Bulk逐项结果、refresh、ID/正文与向量数量后传入发布结果，
已完成真实ES部分失败、版本隔离与重启验证。

旧版本保留READY表示仍可回滚，是否当前生效由active_versions唯一行决定。
构建失败记录在任务上，版本保持STAGING供幂等重试，不污染旧的激活版本。
soft_delete_paper 清除激活指针并取消未完成任务，保留版本及块用于审计。
get_version/get_blocks 是可读取历史版本的内部接口；线上检索必须先检查
get_paper/get_active_version，不能把历史读取接口直接当作可见性校验。
当前不提供恢复软删除论文或物理删除文件的入口。

## 验证

在项目目录运行：

    $env:PYTHONPATH='src'
    .venv/Scripts/python.exe -m unittest discover -s tests -p 'test_rag*.py' -v

覆盖数据库重开、幂等导入、配置版本、块损坏、状态机、进度、失败恢复、
租约和旧Worker隔离、两个进程抢占、发布回滚、版本保留及软删除。

## 第四步：BM25 基线

已实现 ElasticsearchPaperIndex，提供 strict mapping、幂等 Bulk、BM25、目录激活版本和来源过滤、可搜索数量检查与旧版本清理。详见项目根目录 docs/bm25-baseline.md；可运行 scripts/bm25_baseline.py --start-es 复现真实 ES 验收。

## 第八步：向量索引

ElasticsearchVectorIndex 扩展正文索引，使用模型配置指纹绑定独立物理索引。VectorSearchService绑定本地E5模型；vector_pipeline负责批量编码、写入核验与SQLite发布。运行 scripts/vector_search.py import/query 使用本地持久数据，详见 docs/elasticsearch-vectors.md。Agent/UI工具集成和混合召回留待后续步骤。
