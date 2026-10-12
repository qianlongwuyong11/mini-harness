# 第十步：LlamaIndex 适配器

可选依赖：`llama-index-core==0.14.25`（完整依赖由 uv.lock 锁定）。
安装：`uv sync --extra rag-adapter --inexact`。`--inexact` 保留本机已安装的解析与向量化依赖。

## 边界

- `adapters/llamaindex.py` 是唯一导入 LlamaIndex 类型的业务模块。
- `to_node` 将一个 SourceBlock/Chunk/Evidence 转成一个 TextNode，不调用 splitter；完整来源元数据保存在 metadata，排除自动拼接到正文与 embedding 输入。
- `PaperRetriever` 继承 BaseRetriever，显式注入 search 后端与 SearchQuery 范围。现有 HybridSearchService 继续负责模型配置校验、ES 两路过滤、版本快照复核和 RRF。
- `LlamaIndexSearchAdapter.search` 对外仍接收 SearchQuery、返回 SearchResult/Evidence；每次请求独立保存结果，不共享可变查询范围。底层只查询一次。
- `LocalEmbeddingAdapter` 继承 BaseEmbedding，复用注入的 EmbeddingService；E5 前缀、归一化由原编码器负责，不再重复添加。异步方法当前仍执行同步 CPU 编码，不表示后台并行。
- 不使用全局 Settings，不创建默认 LLM、云端 embedding、Agent、Memory、另一份向量库或默认导入流水线。
- 本步未注册 UI/Agent 工具，未实现 read_section/get_evidence；检索桥只要求 search 协议，完整服务能力按后续计划补齐。

## 使用

```python
from mini_harness.rag.adapters.llamaindex import LlamaIndexSearchAdapter
from mini_harness.rag.models import SearchQuery

# backend 是已注入 ES、catalog、embedding 的 HybridSearchService。
service = LlamaIndexSearchAdapter(backend)
result = service.search(SearchQuery('自注意力如何调整感受野？', top_k=5))
# result.evidence 中仍是 mini-harness 自有 Evidence。
```

本步缩小为兼容接口，避免为框架重新实现已经验证的召回逻辑。需要 LlamaIndex 后处理时可通过 PaperRetriever 返回的 NodeWithScore 接入；当前没有额外 reranker。

## 验证

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/llamaindex_verify.py
```

2026-10-12：128 项测试，127 通过、1 跳过。新增 4 项覆盖结构文本、来源元数据、请求过滤/Top-K、空结果、分数、本地 Embedding 注入和异常传播；使用假的 search 后端，不需要云端密钥。
真实本地 ES + E5-base：固定 12 个问题，适配前后的完整 SearchResult 均相等（含顺序、文本、来源、分数与候选数），见 llamaindex-adapter-report.json。这证明接口兼容，不代表检索准确率提高。

实现参考：[LlamaIndex BaseRetriever 官方源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py)。
