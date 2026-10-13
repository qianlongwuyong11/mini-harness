# RAG安装、启动、重建与故障恢复

本机单用户部署，不使用Docker。先阅读[最终验收报告](rag-acceptance.md)：功能闭环可用，但留出检索效果和当前系统内存余量未达到本轮门槛。

## 安装与启动

在仓库根目录使用Python 3.11，按pyproject锁定的可选依赖安装：

```powershell
uv sync --extra rag-parser --extra rag-embedding --extra rag-adapter
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/local_es.py install
.\.venv\Scripts\python.exe scripts/local_es.py serve
```

已有虚拟环境、ES安装和模型缓存时不需重新安装。模型缓存准备按[本地Embedding](local-embedding.md)和[解析评估](../scripts/docling_sparsebev_eval.py)的既有流程；Embedding推理只读取本地权重，不静默联网下载。首次Docling可能准备模型，请预先完成解析烟测。缺少依赖/模型时页面会报初始化失败。

另开终端双击 `start-ui.cmd` 或运行 `python -m mini_harness.ui`。默认8765，已占用则使用 `--port 8766`。浏览器刷新不会升级旧服务进程：更新代码后应在原窗口Ctrl+C、等待保存退出后重启。不要强停用户的其他实例。

UI使用的catalog/root/index必须成套一致，不能让新catalog指向旧索引而假设已经自动导入。独立UI验收库启动命令见[UI说明](../UI.md)。上传→等待READY→勾选论文→提问→核对引用原页。默认未勾选不搜索整个库。

## 16GB机器资源建议

- 每次只运行一个模型工作进程，使用UI的共享Runtime；避免额外运行CLI导入和评估模型进程。
- 导入与查询目前串行，长导入期间查询可能等待。关闭不用的其他重型应用后再测量，不能根据一次成功导入认定没有内存压力。
- ES堆2GiB不是总内存上限；模型使用CPU不占用4GB显存，但会占用运行内存。
- 降低Embedding batch可减小部分峰值，不能消除模型权重和解析器占用。改变配置后必须重新做资源验收，不在本轮未经测量直接承诺改善。
- 本轮未自动停止用户应用、调整系统页面文件或修改ES堆。最低可用内存约200MiB，资源验收未通过。

## 重建与恢复

| 情况 | 处理 |
|---|---|
| ES未连接 | 先运行 `scripts/local_es.py health`，检查本地服务和connection.json；不要关闭TLS验证。队列保留，服务恢复后处理 |
| 导入失败 | 保留错误阶段及产物，在UI点重试，或 `scripts/ingest_pdf.py retry job-ID`，再让同catalog的worker运行 |
| 程序异常退出 | 使用相同配置启动，等待最长约300秒租约到期后恢复；不要删除锁文件来抢任务 |
| 取消任务 | UI取消撤销所有权，当前外部调用结束后停止后续发布；不是立刻强杀解析线程 |
| 文件缺失/摘要错误 | 从备份恢复对应文件后重试；不能只把SQLite状态改成READY |
| 更新解析/分块/Embedding配置 | 采用新的catalog/root/index全量导入并验收后再切换启动参数，保留旧库；不要向原向量字段混写不同维度/版本 |
| 仅删除ES索引 | READY任务不会自动重建索引；使用新独立catalog/root/index重导原PDF，不能仅再次上传同hash期待触发重建 |
| 会话被另一个进程使用 | 同一JSONL会话避免CLI/UI同时恢复；当前会话后端仍无跨进程写入保护 |

安全的独立重建示例（名称自行取新值）：

```powershell
.\.venv\Scripts\python.exe scripts/ingest_pdf.py import "C:\path\paper.pdf" --catalog .mini-harness/rag/rebuild.sqlite3 --root .mini-harness/rag/rebuild --index mini-harness-papers-rebuild
.\.venv\Scripts\python.exe -m mini_harness.ui --port 8767 --rag-catalog .mini-harness/rag/rebuild.sqlite3 --rag-root .mini-harness/rag/rebuild --rag-index mini-harness-papers-rebuild
```

备份应在停止写入后同时保留SQLite（或使用SQLite backup API）、`.blocks`目录、原PDF及解析/向量产物；ES数据用受支持的snapshot或由原PDF重新建立索引，不能复制正在运行的data目录当成可靠备份。ES snapshot自动化尚未实现。模型配置和版本必须一起保留。凭据文件和个人批注不要提交到Git。

## 能力边界

普通Docling不保证公式LaTeX准确；章节归属可能受复杂布局影响；图注不等于图片理解。当前中文关键词扩展偏向SparseBEV，留出集证明混合检索不总优于向量；不要将开发集分数当泛化效果。回答引用校验只确认来源结构，仍需人工核对语义。移出论文库保留文件，磁盘垃圾回收暂未实现。端到端聊天模型延迟/大规模并发/多论文系统评估未完成。
