# Agent 基准数据

生成器生成 4 个回归用例与 3 个待审 heldout 用例，并绑定实际活动论文版本。它们是人工审定前的草稿，不是已验证的标准答案。

包括 66f807 的空范围/旧 ID 场景、中文创新点检索、多轮取消选择、证据不足、局限分析、要求扩大范围。历史依据：turn10 四次工具返回 `Paper outside the UI-selected scope`；turn11 `list_papers` 返回空列表。重构场景不等于原会话完整重放。

生成的带本机论文 ID 的数据建议放入 `.mini-harness/evals/benchmarks/`；见 `docs/agent-eval-runbook.md`。不要用已经反复调试的回归问题冒充独立盲测题。
