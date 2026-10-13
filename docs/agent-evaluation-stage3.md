# 阶段 3：数据集与持久化评测任务

已实现执行基础设施；规则评分、DeepEval 语义评分和评测 UI 留待后续阶段。

## 核心代码

- `evaluation/dataset.py`：读取 UTF-8 JSON/JSONL，校验严格契约、重复字段、文件大小、路径范围；按 paper_id + 活跃 version + source_hash 绑定论文。文件最多 8 MiB，不按标题猜测论文。
- `evaluation/store.py`：SQLite 保存数据集快照、运行配置、任务状态、各次 attempt 与结果引用。默认单 worker，操作系统文件锁避免两个进程同时执行/恢复同一队列。
- `evaluation/runner.py`：独立 Session 驱动真正的 Agent loop，复用共享 PaperRuntime，挂载原有重试、checkpoint、pruner、spill、RAG 引用校验插件。仅注册三个论文只读工具；普通聊天的 shell、文件写入、子 Agent 等不开放。这是明确的 `paper-readonly-v1` 评测配置。
- `evaluation/__main__.py`：校验、入队、查看、执行、取消与重跑 CLI。
- `agent.py`：步数上限变为构造参数（默认仍为 10）；模型和工具追踪附带 run/case/attempt ID。

## 数据集格式

JSON 使用阶段 0 的 `EvaluationDataset` 契约。JSONL 第一行是 manifest（schema_version、dataset_id、papers，不含 cases），后续每行一个 EvaluationCase。两种格式校验后使用相同的规范化摘要。

每个 turn 的 `selected_papers` 使用 manifest 的 logical_key；空数组明确表示没有选中论文。版本或源文件摘要不一致会在调用模型前失败，执行边界也会复查版本。`gold_evidence` 的源摘要必须与绑定一致；块/页是否命中留给评分阶段。

只把 `turn.input` 和映射后的论文选择送入 Agent。scenario、expected_behavior、reference_answer、key_points、gold_evidence、required_tools/forbidden_tools 都保存在评测侧；工具约束后续用于评分，不能通过禁止预期错误工具来人为提高评分。

每例一个新 Session/证据账本；同例多个 turn 共用 Session。不同例共享昂贵的 RAG 模型缓存，不共享历史。

## 持久化与生命周期

默认目录 `.mini-harness/evals/`：

```text
evaluation.sqlite3
worker.lock
<run_id>/<case_id>/<attempt_id>/
  trace.jsonl
  artifacts/<hash-prefix>/<sha256>.json
  spills/...
```

SQLite：
- datasets：规范化数据集摘要与不可变快照。
- runs：配置快照/摘要、排队或运行状态、原因、时间、质量 verdict。
- case_attempts：独立 attempt/session ID、状态、目录、报告引用及错误类型。

请求、输出与报告使用内容寻址的 TraceStore，先写临时文件并 fsync、原子改名，再提交报告引用。trace.jsonl 在现有 checkpoint、安全结束边界保存，移除推理字段并脱敏。进程突然退出可能丢失最近一个 checkpoint 之后的事件，不能宣称零丢失。SQLite/产物仍可能留下未引用文件，本阶段不自动删除。

`completed` 只代表执行结束；verdict 保持 incomplete，report.scoring_status 为 pending，工具失败也不直接等于执行崩溃。评分将在阶段 4/5 接入。

取消复用既有契约：run 最终为 interrupted、reason=cancelled；未开始的任务不会发出模型请求。已开始的调用在返回后于安全边界停止。停止 worker 不自动删除任务。只有持有 OS 锁的 worker 才能将上次崩溃留下的 running/cancelling 标记为 interrupted，不会抢活跃进程任务。

retry 给失败/中断例创建新 attempt，保留旧结果，跳过此前完成例；不会从半个模型调用继续。如果需要所有例重新执行，应重新 enqueue。同一 run 的数据集快照不随源文件修改。

## 预算与当前边界

- 每 turn 的模型/工具循环步数 1–100 可配，重试/引用修复计入独立请求与 token。
- 每例 timeout_seconds 是安全边界截止时间，不是强制杀线程；同步模型调用、ES、向量编码等正在运行时需等待返回。现有 DeepSeek 客户端 HTTP 超时为 60 秒，但这不等于整例硬性 60 秒墙钟上限。
- token 预算按 provider usage 累计，达到限额不再发起下一次请求；最后一次响应可能超过预算，届时标记失败并保留实际用量。配置 token 上限而 provider 不提供 usage 时失败，不能按零计算。
- 默认单队列 worker、单例执行，不承诺分布式共享 SQLite。
- 只支持 live_rules 模式的执行部分；live_judge/rescore 显式拒绝，未伪装成完成评分。
- 版本校验不代替 ES/模型健康检查，运行时检索错误仍会进入真实工具轨迹。正式 UI 的预检/授权提示在阶段 6。
- 模型/工具 schema 和输入输出保存在实际 trace 中；本阶段的运行配置快照尚不包含完整 Git/依赖/索引配置指纹。
- 追踪 JSON/会话日志脱敏，但继承的 spill 插件文件属于受限本地原始工具产物，不能直接作为已脱敏导出。

## 命令行使用

在项目根目录 PowerShell 执行（本地源码需设置 PYTHONPATH）：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m mini_harness.evaluation validate <数据集目录> <相对文件名.json>
.\.venv\Scripts\python.exe -m mini_harness.evaluation enqueue <数据集目录> <相对文件名.json>
.\.venv\Scripts\python.exe -m mini_harness.evaluation status <run_id>
```

入队不会调用远程模型。确认本队列的数据可以发送给当前模型服务后，再执行：

```powershell
.\.venv\Scripts\python.exe -m mini_harness.evaluation work --allow-model-api
.\.venv\Scripts\python.exe -m mini_harness.evaluation cancel <run_id>
.\.venv\Scripts\python.exe -m mini_harness.evaluation retry <run_id>
```

work 清空当前待执行队列后退出；cancel 在另一个终端执行。后台集成提供 EvaluationWorker.start()/close()，UI 尚未接入。自定义 catalog 时 enqueue 和 work 都传同一个 --catalog；worker 不允许默默换库。

## 验证

新增测试覆盖 JSON/JSONL、路径限制、重复字段、版本不匹配、标准答案隔离、多轮连续/跨例隔离、RAG 插件只读工具、取消与新 attempt 重跑、跳过完成例、OS 排他锁、中断恢复、步数/token 预算、落盘失败不重复模型请求。

全部使用本地假模型与临时 catalog，不发送论文、问题或答案到远程 API。
