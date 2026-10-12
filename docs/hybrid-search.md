# 第九步：BM25 + 向量混合检索

2026-10-12 完成。入口 `HybridSearchService.search(SearchQuery)` 返回自有 SearchResult/Evidence；`search_with_trace` 额外返回查询改写、版本范围、两路候选、各路排名、融合分数及阶段耗时。

## 执行过程

1. 从SQLite读取一次已发布版本列表，按论文范围和模型ID/revision筛选，供两路共享。
2. 原始中文问题通过明确的中英领域词表扩展，例如“感受野”补充 `receptive field`；扩展仅送入BM25。记录 original_query、bm25_query、matched_terms。
3. 原始问题由E5-base编码后进行ES kNN。BM25与kNN顺序执行，各取默认20条候选。
4. 两路在ES端使用相同的论文/版本、source_kind、embedding_profile过滤。查询结束后再次检查SQLite，去除已被删除或切换的版本。
5. 按chunk_id合并，使用加权RRF：`bm25_weight/(rrf_k+bm25_rank) + vector_weight/(rrf_k+vector_rank)`；未命中某一路，该路贡献为0。默认k=60、两路权重各1，同分按chunk_id排序。

每条Evidence保留bm25_rank、vector_rank，trace.fusion还保留每一路分数贡献。使用排名融合，不直接相加BM25与ES向量原始分数。同一路重复ID只计一次，相同ID内容冲突时报错。任何检索通道失败均显式报错，目前没有静默降级为另一条通道。

词表位于 `src/mini_harness/rag/hybrid.py` 的 TERMS，是针对三维感知论文的可编辑初版，并非通用翻译系统；未知术语仍由原问题的向量检索处理。词表编写参考了现有论文及问题用语，因此这组题是开发集，不是独立测试集。

## 同一索引、同一12题的比较

| 方案 | Recall@5 | Hit@5 |
|---|---:|---:|
| 原问题BM25 | 29.17% | 33.33% |
| 术语扩展BM25 | 100% | 100% |
| 原问题向量 | 70.83% | 75% |
| 扩展BM25 + 原问题向量，RRF | 93.06% | 100% |

所有查询使用106条作者论文chunk，排除用户批注。指标定义与第七步一致。本轮BM25查询content、title、section并使用现有字段加权；第七步临时基线仅查询content，因此当时20.83%的BM25 Recall@5不能直接作为同配置结果对比。

混合超过原有纯向量基线，但并非本轮最优Recall：扩展BM25单路更高。q4的相关块66和q7的相关块62在融合Top5中被挤出，体现向量路引入的排名干扰。未为追求100%继续针对这12题调整权重，也未添加reranker。下一轮应增加新论文与未参与词表编写的问题，再评估权重、候选数、词表覆盖和章节标题处理。

12次查询总耗时中位约153.87ms，包含BM25、query编码、ES kNN、目录检查、RRF及诊断构建，排除模型启动、LLM生成和UI。只有一轮小样本，不能视为稳定吞吐或延迟承诺。逐题候选与分数见 `docs/hybrid-search-report.json`。

## 使用

先启动已有本地ES，在项目目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/hybrid_search.py "自注意力如何调整感受野？" --top-k 5
.\.venv\Scripts\python.exe scripts/hybrid_search.py --evaluate
```

可选参数：`--candidates 20 --rrf-k 60 --bm25-weight 1 --vector-weight 1`；`--no-expansion`关闭术语扩展。结果JSON包含引用正文、页码和诊断信息。当前CLI每次启动仍需加载模型；后续Agent常驻服务复用编码器。

该步骤复用现有ES索引、E5-base和已发布版本，不需要重新解析、编码文档或重建索引。第十步再封装框架适配器。
