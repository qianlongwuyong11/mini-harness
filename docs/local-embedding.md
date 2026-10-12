# 第七步：本地 Embedding 选型与实测

2026-10-12：选定 `intfloat/multilingual-e5-base`，CPU、4 线程、batch size 4、float32、768 维。模型前向输出按 attention mask 做 mean pooling，随后 L2 归一化。文档加 `passage: `，问题加 `query: `。超过 512 tokens 明确报错，不静默截断。

## 候选与选择依据

| 模型 | 许可证 | 维度 | 输入上限 | 结论 |
|---|---|---:|---:|---|
| multilingual-e5-small | MIT | 384 | 512 | CPU 更快，但本篇样本的中文检索召回偏低 |
| multilingual-e5-base | MIT | 768 | 512 | 本轮选中，质量收益优于增加的内存与延迟 |
| paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 384 | 推荐配置 128 | 仅查阅模型卡，未下载实测；输入较短，不直接适配现有块 |

官方资料：[E5-small](https://huggingface.co/intfloat/multilingual-e5-small)、[E5-base 固定版本](https://huggingface.co/intfloat/multilingual-e5-base/tree/d128750597153bb5987e10b1c3493a34e5a4502a)、[MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)。推理使用 PyTorch 和 Transformers，无需引入 SentenceTransformers 框架。

## 本机测量

共同语料为第六步 106 个作者论文 chunk，排除 21 个用户批注；标题和图注仍在候选中。使用 `docs/embedding-queries.json` 的 12 个手工设计问题：8 个纯中文，4 个带英文术语。人工相关性标签在查看检索排名前建立；只对解析文本核查，尚未由用户复核，也不构成独立测试集。没有根据实验结果修改标签。

| 指标 | E5-small | E5-base | ES BM25（english analyzer） |
|---|---:|---:|---:|
| Recall@5 | 31.94% | 70.83% | 20.83% |
| Hit@5 | 41.67% | 75.00% | 25.00% |
| 纯中文 Recall@5 | 29.17% | 75.00% | 0% |
| 中英混合 Recall@5 | 37.50% | 62.50% | 62.50% |
| 新进程加载缓存权重 | 13.12s | 9.07s | 已启动服务，无冷启动测量 |
| 106 块编码 | 7.84s | 26.26s | Bulk+refresh 1.08s |
| 单问题编码中位数 | 13.73ms | 45.76ms | HTTP 查询中位数 5.83ms |
| 单问题编码 P95 | 16.15ms | 58.30ms | HTTP 查询 P95 9.19ms |
| Python 进程采样峰值 RSS | 1186MiB | 2141MiB | 客户端约40MiB，不包含 ES 服务 |

Recall@K 为每题“前 K 个命中的标注相关块数 / 该题全部标注相关块数”，再取平均；Hit@K 为至少命中一个标注相关块的问题占比。两者不能混用。可查三个 `embedding-eval-*.json` 的每题排名。

向量检索采用 NumPy 全量点积：small/base 中位分别约 0.047/0.073ms；这是 106 条数据的精确检索，不代表 ES kNN 的延迟。问题计时为三轮共36次，样本量很小。所有模型使用同一语料，无跨语言查询翻译或额外重排序。

内存每50ms采样 Python RSS，含模型、依赖和批处理。BM25 的 ES 服务末尾 JVM 已用堆约298MiB，配置堆上限2GiB；这不是 ES 的进程峰值 RSS，不可与模型 RSS 直接等同比较。模型加载指标包含 Python 推理库导入与缓存权重加载，受操作系统缓存影响，base 更快不说明它固有冷启动更快。权重下载不包含在推理耗时中。

结论仅用于当前硬件和单篇论文的初选。base 仍有3题 Top5 未命中，不能宣称通用论文问答达标。small 排名中有较多短标题；后续应评估标题作为检索结果或上下文元数据的不同策略。256/384 分块预算对照、多篇论文及用户真实问题是后续质量优化项。

## 实现与版本

- `rag/embedding.py` 提供 `LocalE5Embedding`，遵循已有 `EmbeddingService` 方法。
- `rag/embedding-default.json` 固定默认配置，revision 为 `d128750597153bb5987e10b1c3493a34e5a4502a`。
- `load_default_config()` 读取该配置；将下载的模型目录传给 `LocalE5Embedding(config, model_path)` 即可使用。
- 模型加载校验下载元数据的 commit，离线加载 safetensors，不执行远程模型代码。
- 默认使用 CPU。`device=auto/cuda` 在 CUDA 不可用时回退 CPU；加载或推理 CUDA OOM 时迁移 CPU 并重试。此机当前是 CPU 版 PyTorch，无 CUDA 路径已实测，OOM 路径使用单元测试模拟，未作真实 GPU 验收。
- 向量范数实测约0.99999988～1.00000012。批处理与单条编码、实际 tokenizer 与分块计数一致；超限输入拒绝执行。
- 114 项回归测试：113 通过，1 项 Windows 环境测试跳过。

重新使用 base tokenizer 分块，生成新的 staging 版本 `557c3224b3e11dd5d1372e20`。两个 tokenizer 文件指纹不同，但本篇正文分块文本完全一致；检查所有模型输入逐条相等后，复用本轮 base 向量。旧版产物保留。

`docs/embedding-selected.json` 和本地 `selected/<version>/manifest.json` 保存模型 revision、维度、tokenizer SHA-256、池化、归一化、前缀及 chunk ID 列表。`version.json` 同时记录 parser、chunker 和 embedding 版本。`embeddings.json` 保存 chunk_id → vector，106 条作者证据已编码，21 条批注保留但未编码。产物尚未导入 ES，也未向 SQLite 发布 READY，第八步基于该版本建设向量索引。

## 复现

依赖归入可选 `rag-embedding` extra，并更新 uv.lock。当前已安装 CPU PyTorch；新环境若 uv 默认解析到 GPU 构建，应遵循 PyTorch 官方 CPU 安装指引，避免下载不需要的 CUDA 依赖。

下载固定权重使用 `huggingface_hub.snapshot_download`，repo_id 与 revision 取 `scripts/embedding_eval.py` 的 MODELS；local_dir 设为 `.mini-harness/rag/models/<模型短名>`，allow_patterns 为 config.json、tokenizer.json、tokenizer_config.json、special_tokens_map.json、model.safetensors、sentencepiece.bpe.model。保留 `.cache/huggingface/download` 的版本元数据。

在项目目录分别运行：

```powershell
.\.venv\Scripts\python.exe scripts/embedding_eval.py small
.\.venv\Scripts\python.exe scripts/embedding_eval.py base
# 另一个终端启动已有本地 ES 后运行：
.\.venv\Scripts\python.exe scripts/embedding_eval.py bm25
.\.venv\Scripts\python.exe scripts/embedding_verify.py
.\.venv\Scripts\python.exe scripts/embedding_prepare_selected.py
```

BM25 脚本只创建随机命名的临时测试索引，结束后删除该索引；既有索引不变。两个模型分别运行于新进程，权重驻留内存不会叠加。下载目录占用约1.6GB，后续正常运行只需加载选中的 base。
