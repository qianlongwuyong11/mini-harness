# 第15步：RAG质量与资源验收

日期：2026-10-12。评估代码和本轮测量已完成；**综合验收未通过，不能标为质量及资源全部达标**。未根据留出结果修改术语表或融合参数。

## 固定数据与指标

`rag-eval-dataset.json`固定12个开发问题、8个新中文留出问题。开发问题已用于E5/混合检索选型；留出问题在本轮首次运行前标注。每题记录chunk_id、内容SHA256、页码和理由，整份语料也有SHA256。标签由助手读取原文后标注，尚非独立人工金标准。所有问题来自SparseBEV，不能代表跨论文能力。

Recall@K：命中标注块数/全部标注块数，逐题宏平均；Hit@K：至少命中一块的问题比例；MRR@K：前K名中首个正确块排名的倒数。BM25原文查询与术语扩展版分开报告，向量保留中文原query，混合使用扩展BM25+向量RRF。

| 数据集 | 检索方式 | Recall@5 | Hit@5 | MRR@5 |
|---|---|---:|---:|---:|
| development | bm25 | 29.17% | 33.33% | 0.250 |
| development | bm25_expanded | 100.00% | 100.00% | 0.736 |
| development | vector | 70.83% | 75.00% | 0.562 |
| development | hybrid | 93.06% | 100.00% | 0.778 |
| heldout | bm25 | 25.00% | 25.00% | 0.067 |
| heldout | bm25_expanded | 31.25% | 37.50% | 0.192 |
| heldout | vector | 56.25% | 62.50% | 0.458 |
| heldout | hybrid | 43.75% | 50.00% | 0.417 |

预先在评估脚本设定的检索门槛：留出Hybrid Hit@5≥75%，Recall@5不低于纯向量；两项均未通过。8题样本很小，4题命中不等于稳定50%的产品准确率。

失败分析：Hungarian匹配、优化器超参数、解码器层数/共享权重的问题，两路Top5均未命中标注实现细节块；NDS指标问题向量命中但融合后跌出Top5。中文查询对英文BM25较弱，现有术语表围绕方法模块，对实验设置覆盖不足。下一轮应在开发集比较查询改写、标题过滤、分块和融合权重，再另收独立论文的新验收集；本批留出题此后视为已观察集。

## 在线延迟（不含聊天模型）

| 环节 | 中位数ms | P95 ms |
|---|---:|---:|
| encoding_ms | 42.95 | 55.44 |
| bm25_ms | 10.42 | 14.29 |
| knn_ms | 16.05 | 22.93 |
| total_ms | 80.76 | 105.84 |

新进程导入推理库并加载缓存模型约10.33s；首个检索约90.55ms。上述延迟包含query编码、ES关键词/kNN和融合，20题各执行一次；各阶段中位数不能直接相加得到整体中位数。操作系统文件缓存未清空，不是磁盘冷启动；也不是UI首次启动时间。

## 完整导入资源

独立catalog和索引导入14页SparseBEV，106个正文chunk；模型初始化14.85s，之后完整导入86.00s，摊销每页6.14s（不是逐页计时）。

| 阶段 | 约耗时s |
|---|---:|
| startup | 16.30 |
| parsing | 51.13 |
| chunking | 0.49 |
| embedding | 30.46 |
| indexing | 3.10 |
| publishing | 1.09 |

Python进程树峰值RSS 2.77 GiB，不含ES；系统最低可用内存199.7 MiB，未达到预设2 GiB余量。系统页面文件使用量从3.90增加到5.33 GiB。系统同时运行其他应用，不能将变化全归因于本任务；Windows计数不足以量化换页频率。没有发生本次任务被杀，但无法确认“不频繁换页”。

本轮本地产物约11.25 MiB；ES段文件0.46 MiB，translog另有1.98 MiB。段文件+translog只是当前索引统计，不含ES安装、模型、日志等总占用，也不能线性推导大库容量。

## 回答复核与人工验收

复用已授权生成的四类DeepSeek回答，本轮没有新模型外发。脱敏复核在 `rag-answer-review.json`，完整答案/用户批注只留在忽略目录 `.mini-harness/rag/answer-eval/report.json`。

| 类别 | 助手复核 | 关键问题 | 人工签核 |
|---|---|---|---|
| 方法 | 部分通过 | 三模块有支持，但目录状态不应引用论文图注证明 | 待用户复核 |
| 上下文 | 部分通过 | 参数解释有支持，公式文本未精确核验、部分章节归属不准确 | 待用户复核 |
| 批注 | 通过 | 引用user_annotation并明确与作者观点分离 | 待用户复核 |
| 无答案 | 未通过 | 已拒答但“论文中仅有”夸大了片段覆盖范围 | 待用户复核 |

引用ID与页码存在只是结构校验，不证明结论被原文支持。严格答案质量门槛未通过，人工验收仍待完成。人工复核应逐条确认事实与引用、公式是否一致、拒答是否仅限证据范围、批注是否混入作者结论；不得只检查引用数量。

## 默认配置与复现

保留当前基线，尚不称为通过验收的最终推荐：普通Docling/公式增强关闭；结构分块384 token；multilingual-e5-base固定revision、768维、CPU 4线程/batch4/输入≤512/L2；ES单节点2GiB堆、english BM25、HNSW；每路候选20、RRF k=60、权重1:1、top5、证据预算6000。重排序未启用。

```powershell
cd F:\github\mini-harness
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/rag_acceptance.py
.\.venv\Scripts\python.exe scripts/rag_resources.py "C:\Users\yjq\Desktop\Perception\SparseBEV.pdf"
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

检索脚本只读默认vector-catalog及其既有selected语料，检查版本/摘要；不是读取UI验收库。资源脚本每次创建随机独立目录/索引，保留测量产物供追溯，不删除用户数据。重复执行会增加磁盘占用。模型必须已下载，ES必须健康，未就绪应先按照运行手册修复。

详细报告：`rag-acceptance-report.json`（每题排名/延迟），`rag-resource-report.json`（原始资源数值），`rag-answer-review.json`（答案问题）。运行方式见 `rag-runbook.md`。
