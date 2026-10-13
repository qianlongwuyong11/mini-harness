# 阶段 0：依赖与评测契约

2026-10-13 完成。已具备契约与隔离 SDK 验证入口；尚未实现数据集运行器、语义评分适配或评测 UI。

## 依赖决定

DeepEval 固定 4.2.8，Pydantic 固定 2.14.0。Python 3.11.16 / Windows 已验证。

完整安装检查发现 DeepEval 的 click>=8,<8.4 与当前 huggingface-hub 的 click>=8.4.2,<9 冲突。因此评分 SDK 使用独立 `.venv-eval`，不安装到 Agent 的 `.venv`。原环境已按安装前快照恢复，click=8.5.0、rich=15.0.0、tabulate=0.10.0；LlamaIndex=0.14.25 保留。

- 原环境：运行 Agent、Docling、E5、ES、LlamaIndex 检索评分。
- 评分环境：读取未来导出的 JSON 轨迹，执行 DeepEval 评分，返回结构化结果。
- `eval-contracts` 可选 extra：只安装 Pydantic，用于契约验证，不导入 DeepEval。
- `requirements/agent-eval.in`：评分直接依赖；`.lock`：当前 Windows/Python 3.11 全量版本和包哈希，其他平台需重新验证。

复现（项目根目录，uv 可用）：

```powershell
uv venv --python .venv/Scripts/python.exe .venv-eval
uv pip sync --python .venv-eval/Scripts/python.exe --require-hashes requirements/agent-eval.lock
.\.venv-eval\Scripts\python.exe scripts/verify_eval_stage0.py
```

SDK 进程通过 `worker_environment` 构造独立环境，不修改父进程；屏蔽自动 dotenv 和遗留密钥文件读取、遥测、升级检查，关闭 SDK 自写文件。未来报告由 mini-harness 存储。已有 SDK 提前加载或有托管平台配置时拒绝继续，避免开关设置过晚。不得直接调用托管登录、上传或自动 instrument；阶段 5 的语义指标必须显式提供 judge adapter。

这不是“永远不联网”的保证：阶段 5 配置远程 judge 后会按授权发送评测内容。当前验证只用无网络 fixture，没有发送论文、批注或用户对话。

## 文件职责

| 文件 | 内容 |
|---|---|
| evaluation/models.py | 数据集/论文绑定/用例、trace ID、产物引用、指标结果、运行状态 |
| evaluation/registry.py | 统一指标目录、依赖数据、单位、方向、计算方和结果校验 |
| evaluation/backend.py | 固定 SDK 版本、本地模式、子进程环境、显式 judge 约束 |
| scripts/export_eval_contracts.py | 从 Python 类型重新导出 JSON Schema 和指标目录 |
| scripts/verify_eval_stage0.py | 网络连接禁止下验证 SDK 导入、工具指标、dotenv/旧密钥隔离和文件不变 |
| docs/evaluation/v1/ | 5 份结构 schema 与可供未来 UI 使用的 metrics.json |

契约拒绝未知字段、重复 case ID、未绑定论文、缺失父级 trace、路径穿越、NaN/Inf 和将缺失指标伪记为零。selected_papers=[] 明确表示无选择。标准答案和 gold 属于评测输入，后续运行器必须仅投递 turns 给 Agent。JSON Schema 表达结构限制，跨字段检查仍以 Python 校验器为准；路径契约检查不替代阶段 3 的实际磁盘路径/重解析点检查。

指标目录是定义，不代表全部已实现。`implementation=existing` 仅指原有统计/检索能力存在，新结果格式适配留待后续。语义指标阈值留空；范围/引用违规计数门槛为零。禁止将原始检索分数展示成校准概率。MetricResult 的 passed 与 threshold 必须成对解释，错误/未适用/缺数据单列。

## 验证结果

- 原环境 140 个包：uv pip check 通过；隔离评分环境 67 个包：通过。
- 契约测试 7 项通过，覆盖数据引用、缺失分数、状态、路径及父子环境隔离。
- SDK 4.2.8 工具指标：正确调用 1.0，缺失调用 0.0。
- 验证窗口中 socket 连接尝试 0；SDK 未读 fixture 云端 key、未新增或改写文件。
- 阶段 0 不构成 Agent 效果验收，既有 RAG 质量门槛未通过的结论不变。

后续阶段 1 完善实际模型请求和 step 轨迹；阶段 3 建运行器；阶段 5 从主环境启动独立评分进程，衔接两端契约。
