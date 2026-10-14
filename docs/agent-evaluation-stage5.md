# 阶段 5：DeepEval 语义评分与独立重评分

## 实现

- `scoring.py`：校验明确的 judge 配置，从已完成 attempt 的 trace 提取每轮最终回答与该次模型请求实际注入的证据；启动独立 `.venv-eval` 子进程。主 Agent 环境不导入 DeepEval。
- `judge_worker.py`：使用阶段 0 的本地配置启动锁定的 DeepEval 4.2.8，仅调用 metric.measure；不登录、不调用托管上传/自动评测运行 API。
- `judge.py`：标准库 HTTP 的 OpenAI-compatible Chat Completions judge。明确模型、地址和密钥环境变量，固定 temperature=0；支持 JSON schema 校验、请求数/输出 token 上限、请求超时。
- `deepeval_adapter.py`：单轮用 LLMTestCase + GEval，多轮用 ConversationalTestCase + ConversationalGEval；工具名称匹配用 ToolCorrectnessMetric。固定 rubric 版本 `paper-agent-v1`，不临时生成评分步骤。
- `store.py`：新增 score_attempts 表，每次重评分生成 score_id，保存配置、开始/结束时间、状态和报告引用。旧报告不覆盖。
- `runner.py` / `__main__.py`：支持 live_judge 自动评分和对已有运行的 rescore。Agent 执行与 judge 状态分别记录。

## 指标

支持任务完成、回答相关性、参考答案正确性、关键点覆盖、注入证据忠实度、工具名称正确性。评分理由和分数来自明确配置的 judge，不是校准概率。

没有 reference_answer、key_points 或实际注入证据时，对应指标是 missing_data。工具评分只按 required_tools 比较名字，不要求唯一序列、不评估参数语义；forbidden_tools 继续由阶段 4 规则处理。没有 expected tool 名称则该指标缺失。

多轮按独立会话 rubric 评分，每轮证据挂在自己的 assistant turn，不能把其他轮证据混作该轮来源。case 级参考答案视为该会话的预期结论；需要逐轮独立金标时应拆用例或扩展后续数据集契约。

范围、引用等规则失败优先于语义分数。judge 报告的 semantic_verdict 表示本次所选指标的判定，verdict 加上阶段 4 硬规则。未配置阈值时只给分，不自动称为通过。示例 thresholds 为空，阈值应经过人工样本校准。run 原有 verdict 保留规则口径，最新语义判定读取 score 报告，不能把 run.status=completed 当作质量通过。

## 数据与保存

仅向指定 judge 发送所需问题、最终答案、预期行为/参考答案/关键点以及实际注入证据；标准答案依然不会发给被测 Agent。

产物位于 `.mini-harness/evals/<run>/<case>/<attempt>/scores/<score_id>/`。报告、judge 请求/响应和每次调用记录均经 TraceStore 脱敏后内容寻址保存。调用记录包含 metric_id、request_id、耗时、provider_request_id、usage；不保存认证头或模型隐藏推理。

token 使用 API 返回值，缺失字段保持 null。费用未配置版本化价格表，当前为 null。达到请求上限后剩余指标记录 error；超时/取消会终止评分子进程，已落盘调用产物仍保留，但无法保证汇总被终止请求的费用。重启在独占 worker 锁保护下将未结束评分标记 interrupted。

评分文件发布失败时可留下已有产物，score 状态记 error；SQLite 写入本身失败仍属于存储故障。评分与 Agent 会话有独立生命周期，模型/评测器错误不会重跑 Agent。

## 使用

使用 [judge.example.json](evaluation/judge.example.json) 配置实际模型与地址。密钥从指定**环境变量**读取；评分进程不自动读取 `.env`。可以在已安全设置该环境变量的终端运行，密钥不要写进配置文件。

项目根目录 PowerShell：

```powershell
$env:PYTHONPATH = 'src'
# 已存在运行：只读取已有轨迹重新评分，无 ES/embedding/Agent 调用
.\.venv\Scripts\python.exe -m mini_harness.evaluation rescore <run_id> --judge-config docs/evaluation/judge.example.json --allow-judge-api

# 新实验：先执行 Agent，再评分
.\.venv\Scripts\python.exe -m mini_harness.evaluation enqueue <数据集目录> <文件.json> --judge-config docs/evaluation/judge.example.json
.\.venv\Scripts\python.exe -m mini_harness.evaluation work --allow-model-api --allow-judge-api
.\.venv\Scripts\python.exe -m mini_harness.evaluation status <run_id>
```

`rescore` 选择每个 case 最新且完成的 attempt，为本次配置重新计算所选指标，已有 Agent 轨迹不变。不支持给尚未完成的半个回答判语义分；judge 失败通过 rescore 重试，不使用 Agent retry。

`--allow-judge-api` 表示操作者同意把以上评测材料发送到配置的 judge。应用调用 SemanticScorer 也必须显式传 allow_api=True。

UI 接入仍属阶段 6/7。当前只支持文本证据和 OpenAI-compatible JSON 输出模型；不做视觉 judge，不承诺模型分数已经过人工校准。

## 验证范围

用本机 HTTP 假 judge 驱动真实独立 DeepEval 进程，验证单轮/多轮适配、工具名称评分、缺失标注、重评分保留历史、密钥脱敏、调用 token 记录、live_judge 和取消。不向远程服务发送真实论文、问题或答案；这些测试验证适配与流程，不能证明真实 judge 评分质量。
