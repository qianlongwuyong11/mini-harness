# 第四步：纯 BM25 检索基线

已在 Windows 本机 Elasticsearch 9.5.3 实测；复用第三步的 HTTPS/认证客户端。
此步只使用手写合成片段，不是 SparseBEV PDF 解析结果，也不是完整论文检索质量评估。

## 实现

- `rag/elasticsearch.py`：`ElasticsearchPaperIndex` 实现自有 PaperIndex 协议。默认索引名 `mini-harness-papers-bm25-v1`，创建需显式调用 ensure_index()。
- 一个主分片、零副本；strict mapping；不含 dense_vector。
- title/section/content 为 text + BM25；paper_id/version/chunk_id/source_kind 为 keyword；page_start/page_end 为 integer。
- 原始 Chunk（页码、bbox、父子和邻接关系、原文块 ID 等）保存在不参与索引的 chunk 对象中，返回时转换为领域对象。
- 默认 standard；title/section 权重 2，content 权重 1；同分按 chunk_id 排序。
- 已存在的 mapping 必须一致，否则要求新建物理索引；禁止静默修改分词配置。

## 写入与可见性

- 只接受目录中登记的 staging 版本；验证稳定 chunk_id、所属版本、正文摘要以及批内重复 ID。
- Bulk 按条数（默认 100）和字节数（默认 4 MiB）拆批；index 操作用稳定 ID 覆盖写入，允许失败后重试。
- 使用 refresh=wait_for；逐项检查 HTTP 状态和 error，HTTP 200 不代表全部写成功。部分成功不会回滚 ES，抛出 BulkWriteError，由编排层重试整个未发布版本。
- count_version 检查可搜索数量；索引适配器不会擅自激活 SQLite 版本。最终编排在第 11 步完成。
- 此版本不接受 embeddings；vector_search 明确抛出 NotImplementedError。

## 查询与删除

1. list_active_versions 从 SQLite 单次读取未删除的 READY 激活版本。
2. 查询按 `(paper_id=A AND version=V1) OR (paper_id=B AND version=V2)` 过滤，避免两个独立 terms 造成错误交叉组合。
3. 同时在 ES 侧预过滤 source_kind，默认只查询 paper。
4. ES 返回后再读目录，删除/换版的结果被丢弃；可能少于请求数量，不自动填入过期证据。
5. 查询超时或分片失败直接报错；没有激活论文则返回空，不发送无范围查询。
6. delete_version 拒绝删除当前活动版本；需先软删除论文或切换版本。清理失败可重试。

最后一次目录检查之后仍可能发生状态变更；这里不声称 SQLite 和 ES 构成跨库事务。回答/引用层后续还需要验证版本。当前本地基线最多一次选择 512 篇活动论文。

## 复现

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/bm25_baseline.py --start-es
```

自动复用健康 ES；未运行时启动项目实例，最后只关闭自己启动的实例。分别创建随机命名的 standard/english 测试索引，验证后删除这些测试索引。报告保存在 `docs/bm25-baseline.json`；可用 --report 指定其他输出位置。不会导入实际 PDF 或修改用户论文库。

单元测试：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_rag_elasticsearch.py -v
```

## 实测结果（2026-10-11）

| 查询 | standard Top-1 | english Top-1 |
| --- | --- | --- |
| SparseBEV | Overview | Overview |
| nuScenes | Experiments | Experiments |
| scale-adaptive self attention | Scale-adaptive Self Attention | Scale-adaptive Self Attention |

| 输入 | standard tokens | english tokens |
| --- | --- | --- |
| nuScenes | nuscenes | nuscen |
| scale-adaptive self attention | scale, adaptive, self, attention | scale, adapt, self, attent |
| multi-camera | multi, camera | multi, camera |
| detection | detection | detect |

当前默认 standard，优先保留技术术语词形；english 的词干化也可能帮助词形变体召回，待真实论文固定问题集评估后再决定。当前小样本不证明任何一种总体更优，也不把 BM25 分数解释为概率。

真实 ES 检查全部通过：重复写入数量不增加、暂存版本不可见、论文范围、版本发布切换、旧版清理、批注隔离、软删除不召回。
新增 8 项自动化测试覆盖 Bulk 部分失败/缺失确认、输入不合法、版本过滤和查询期间删除竞态等。完整测试 100 项：99 通过，1 项原有 Windows 符号链接权限跳过。

参考：
- https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-bulk
- https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter

下一步是第 5 步 Docling 结构解析；尚未在 Agent 工具或 UI 中接入搜索。
