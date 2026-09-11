# 知识库治理：子 Agent 并发修复改造方案

> 状态：已执行 ｜ 目标工作区：`F:\github\mini-harness` ｜ 基线版本：`mini-harness 0.3.0`（27 个模型工具 → 28 个）
>
> 实施时与本方案的差异见文末「14. 执行记录」。

## 1. 目标

在不破坏现有插件化结构的前提下，引入"审计—并发调查—串行应用—统一复审"的知识库修复闭环：

- 主 Agent 用 `kb_audit` 一次性拿到确定性事实；
- 子 Agent 并发调查上下文并产出结构化 `ChangeSet`（只读，不写盘）；
- 主 Agent 做冲突检测与串行应用；
- 修改后统一复审，以文件终态而不是模型自述作为验收依据。

### 非目标

- 不让子 Agent 直接写文件；
- 不实现向量检索、OCR、跨进程调度；
- 不改动 `Agent` 主循环（`src/mini_harness/agent.py` 不修改）。

## 2. 关键设计决策

### 2.1 为什么不让子 Agent 并发写盘

| 风险 | 场景 | 后果 |
| --- | --- | --- |
| 写冲突 | 多个 `broken_link` 都在 `index.md` | 后写覆盖先写 |
| 全局决策被拆散 | `duplicate_title` 涉及多篇笔记 | 两个子 Agent 同时改名，仍然重复 |
| 乐观锁失效 | 子 A 读完、子 B 先写 | `FS_STALE_VERSION` 或脏覆盖 |
| 不可验证 | 子 Agent 自述"已修复" | 无法回归审计 |

因此并发只用于**只读调查与方案生成**，写入由主 Agent 串行完成。

### 2.2 为什么现有 `subagent` 工具不够用

`src/mini_harness/subagent.py` 的 `run_subagent()` 只调用 `client.chat(messages)`，不传 `tools`，最多 3 步。子 Agent 无法调用 `read`、`grep`、`kb_search`，只能"用自然语言假装审计"。所以必须新增一种**带受限工具集**的子 Agent 运行方式。

### 2.3 复用而非替换

- 复用 `KnowledgeBaseService.audit()` 的 issue id（`sha256(rule+path+targets)[:16]`，稳定可寻址）；
- 复用 `ToolRegistry` 投影子集，不新建第二套工具体系；
- 复用 `WorkflowEngine`（`ThreadPoolExecutor`，失败项结算为 `None`）做并发；
- 复用 `FilesystemService.edit` 与 `SandboxPolicy`、`ObservationTracker`，保留 read-before-write 与 mtime 校验。

## 3. 总体流程

```text
用户: "修复知识库里的断链和重复标题"
  │
  ├─ 1. kb_audit                    主 Agent，确定性全库扫描
  │
  ├─ 2. kb_repair(dry_run=true)     主 Agent 调用编排工具
  │     ├─ 按 write_paths 分组 issue → RepairTask[]
  │     ├─ 冲突分批（互不重叠的任务同批并发）
  │     ├─ WorkflowEngine 并发运行只读子 Agent
  │     │     └─ 子 Agent: read / grep / kb_search → 返回 ChangeSet JSON
  │     └─ 汇总 ChangeSet，返回预览（不写盘）
  │
  ├─ 3. 用户确认
  │
  ├─ 4. kb_repair(dry_run=false)    主 Agent 串行应用
  │     ├─ 校验路径、old_string 唯一性、content_hash
  │     ├─ 逐个 filesystem.edit
  │     └─ 自动复审 kb_audit
  │
  └─ 5. 报告：扫描数 / 问题数 / 已修 / 失败 / 剩余
```

## 4. 文件清单

### 新增

| 文件 | 职责 |
| --- | --- |
| `src/mini_harness/kb_repair.py` | 领域对象、任务规划、冲突分批、ChangeSet 应用、`kb_repair_provider` / `kb_repair_consumer` |
| `src/mini_harness/tool_subagent.py` | 带受限工具集的子 Agent 运行器（`run_tool_subagent`） |
| `tests/test_kb_repair.py` | 规划、冲突分批、应用、安全边界、插件生命周期测试 |
| `tests/test_tool_subagent.py` | 受限工具集、步数上限、结构化输出解析、失败结算测试 |

### 修改

| 文件 | 改动 |
| --- | --- |
| `src/mini_harness/capabilities.py` | `AGENT_DEFAULTS` 增加 5 个 `kb_repair_*` 配置项 |
| `src/mini_harness/bundle.py` | `enable_knowledge_base` 分支内挂载 `kb_repair_provider` / `kb_repair_consumer` |
| `tests/test_knowledge_base.py` | 工具总数 27 → 28，补 `kb_repair` 断言 |
| `tests/test_document_import.py` | 同上，工具总数断言同步 |
| `eval/kb_governance_eval.py` | 增加 `repair-concurrent` 案例，新增 `--variant repair` |
| `README.md` | 补充修复闭环、配置项、并发语义说明 |

**不修改**：`agent.py`、`cordis.py`、`registry.py`、`workflow.py`、`session.py`、`fs_tools.py`、`sandbox.py`。

## 5. 数据结构

放在 `src/mini_harness/kb_repair.py`，全部 `@dataclass(frozen=True)` 且 JSON 可序列化。

```python
@dataclass(frozen=True)
class RepairTask:
    """一个可独立交给子 Agent 的修复单元。"""
    id: str                      # "task-01"，按分组顺序稳定生成
    rules: tuple[str, ...]       # 本任务覆盖的审计规则
    issue_ids: tuple[str, ...]   # 来自 kb_audit 的 issue.id
    write_paths: tuple[str, ...] # 允许写入的相对路径（冲突检测的键）
    read_paths: tuple[str, ...]  # 建议阅读的相对路径
    instruction: str             # 给子 Agent 的任务描述


@dataclass(frozen=True)
class KnowledgeChange:
    """一次最小编辑，语义对齐 filesystem.edit。"""
    path: str                    # 相对 kb_root
    old_string: str              # 必须在目标文件中唯一出现
    new_string: str
    reason: str


@dataclass(frozen=True)
class KnowledgeChangeSet:
    """子 Agent 的结构化产出。"""
    task_id: str
    changes: tuple[KnowledgeChange, ...]
    issue_ids: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class RepairTaskResult:
    task_id: str
    status: str                  # planned / applied / skipped / failed
    change_count: int
    applied_paths: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class KnowledgeRepairReport:
    dry_run: bool
    batches: int                 # 冲突分批数量
    tasks: tuple[RepairTaskResult, ...]
    applied_changes: int
    audit_before: Mapping[str, int]   # kb_audit summary
    audit_after: Mapping[str, int] | None  # dry_run 时为 None

    def to_json(self) -> str: ...
```

## 6. 实施步骤

### 阶段 1：受限工具子 Agent（`tool_subagent.py`）

```python
def run_tool_subagent(
    client: ChatClient,
    task: str,
    system_prompt: str,
    tools: Sequence[Tool],
    *,
    max_steps: int = 6,
    cancelled: CancellationSignal | None = None,
) -> SubagentResult:
    """独立 Session + 受限工具集的子 Agent；复用 SubagentResult 语义。"""
```

要点：

1. 独立 `Session`，不继承父历史（与 `run_subagent` 一致）；
2. 每步把 `tools` 传给 `client.chat()`，解析 `reply.tool_calls` 并执行，结果以 `role="tool"` 回灌；
3. 未注册工具、参数非法、执行异常一律配对成 `tool` 结果，不中断循环；
4. 达到 `max_steps` 返回 `stop_reason="max-steps"` 并保留部分文本；
5. 全程可被 `cancelled` 中断，复用 `subagent.py` 的 `CancellationSignal`。

**验收**：`tests/test_tool_subagent.py` 用 FakeClient 断言工具被调用、结果回灌、步数上限、异常结算。

### 阶段 2：只读工具投影

在 `kb_repair.py` 中：

```python
REPAIR_SUBAGENT_TOOLS = ("read", "grep", "glob", "kb_search")

def _project_tools(registry: ToolRegistry, allowed: Sequence[str]) -> tuple[Tool, ...]:
    """从主注册表投影只读子集；缺失工具直接报错，避免静默降级。"""
```

**禁止**给子 Agent：`write`、`edit`、`shell`、`web_search`、`web_fetch`、`subagent*`、`workflow`、`ask_user_question`、`kb_import_document`。理由：防越权写入与无限委派。

### 阶段 3：任务规划与冲突分批

```python
def plan_tasks(
    report: KbAuditReport,
    *,
    issue_ids: Sequence[str] | None,
    max_tasks: int,
) -> tuple[RepairTask, ...]:
    """把 issue 按 write_paths 聚合成任务。"""
```

聚合规则：

| 规则 | write_paths | 说明 |
| --- | --- | --- |
| `broken_link` | `issue.path` | 同一文件的多条断链合并为一个任务 |
| `missing_title` | `issue.path` | 单文件 |
| `missing_frontmatter` | `issue.path` | 单文件 |
| `duplicate_title` | `issue.targets` 全部 | 跨文件，必须单任务全局决策 |
| `orphan` | `issue.path` + 入口页 | 需要改 `index.md`，天然与其他任务冲突 |
| `stale` | — | 默认不自动修复（仅报告） |

冲突分批：

```python
def batch_tasks(tasks: Sequence[RepairTask]) -> tuple[tuple[RepairTask, ...], ...]:
    """贪心分批：同一批内任意两个任务的 write_paths 交集为空。"""
```

同批并发，批与批之间串行。`index.md` 这类共享入口页自然落到单独批次。

### 阶段 4：并发执行与 ChangeSet 解析

```python
changesets = workflow.parallel([
    lambda task=task: _run_one(task) for task in batch
])
```

- 并发上限取 `kb_repair_max_concurrency`（默认沿用 `jobs_max_concurrency`）；
- `WorkflowEngine.parallel` 已把单项失败结算为 `None`，不阻塞同批其他任务；
- 子 Agent 输出要求**纯 JSON**，解析失败记为 `status="failed"` 并保留原文到 `error`；
- 解析后立即校验：`path` 必须属于该任务的 `write_paths`，否则整个 ChangeSet 作废。

子 Agent system prompt 要点：

```text
你是知识库修复调查员。只读工具：read / grep / glob / kb_search。
先阅读相关笔记确认事实，再输出唯一一个 JSON 对象：
{"task_id": "...", "changes": [{"path","old_string","new_string","reason"}], "issue_ids": [...], "notes": "..."}
约束：只能修改 write_paths 列出的文件；old_string 必须是文件中唯一出现的原文片段；
不得删除正文、不得重写整篇笔记、不得输出 JSON 以外的内容。
```

### 阶段 5：串行应用与版本保护

```python
def apply_changeset(
    changeset: KnowledgeChangeSet,
    *,
    filesystem: FilesystemService,
    kb_root: Path,
    allowed: frozenset[str],
) -> RepairTaskResult:
```

应用前逐条检查：

1. `path` 解析后必须位于 `kb_root` 内（`relative_to` 校验）；
2. `path` 必须在任务的 `write_paths` 白名单内；
3. 先 `filesystem.read` 目标文件 —— 同时满足 read-before-write 并刷新 `ObservationTracker`；
4. `old_string` 在当前内容中出现次数必须为 1，否则跳过该条并记录原因；
5. 调用 `filesystem.edit`，由 `SandboxPolicy.fence_write` 和 mtime 比对兜底；
6. 任一条失败不回滚已应用的其他条目，但在 `RepairTaskResult.error` 中如实记录。

**应用阶段严格串行**，不使用线程池。

### 阶段 6：工具注册与插件接线

```python
@depends("knowledge_base", "filesystem", "tools", "llm", "workflow", "agent_settings")
def kb_repair_provider(ctx: Context, _config: Any) -> None:
    ...
    ctx.provide("kb_repair", service)


@depends("kb_repair", "tools", "prompt")
def kb_repair_consumer(ctx: Context, _config: Any) -> None:
    ctx.effect(lambda: prompt.section("knowledge-base:repair", ..., order=12))
    ctx.effect(lambda: tools.register(Tool("kb_repair", ...)))
```

工具 schema：

```json
{
  "type": "object",
  "properties": {
    "issue_ids": {
      "type": "array",
      "items": {"type": "string"},
      "uniqueItems": true,
      "description": "来自 kb_audit 的 issue.id；省略表示全部可自动修复的问题"
    },
    "rules": {
      "type": "array",
      "items": {"type": "string", "enum": ["broken_link", "missing_title", "missing_frontmatter", "duplicate_title", "orphan"]},
      "uniqueItems": true
    },
    "dry_run": {"type": "boolean", "default": true},
    "max_tasks": {"type": "integer", "minimum": 1, "maximum": 32}
  }
}
```

`bundle.py` 挂载位置（`enable_knowledge_base` 分支内，`knowledge_watch_provider` 之后）：

```python
ctx.plugin(kb_repair_provider)
ctx.plugin(kb_repair_consumer)
```

> 注意：`kb_repair_provider` 依赖 `workflow`，而 `workflow_provider` 当前在分支之后挂载。Cordis 的 fiber 会在依赖就绪后自动加载，顺序不影响正确性；但需新增测试断言 `kb_repair` 最终处于 `active` 状态。

### 阶段 7：配置项

`capabilities.py` 的 `AGENT_DEFAULTS` 追加：

```python
"kb_repair_enabled": True,
"kb_repair_max_concurrency": 4,
"kb_repair_max_tasks": 16,
"kb_repair_subagent_max_steps": 6,
"kb_repair_auto_reaudit": True,
```

`kb_repair_enabled=False` 时 provider 直接 return，不注册服务与工具（与 `kb_watch_enabled` 的写法保持一致）。

## 7. 提示词改动

`kb_consumer` 现有的 `knowledge-base:governance` 段落保持不变，新增 `knowledge-base:repair`（order=12）：

```text
批量修复时：先 kb_audit 取得 issue.id，再用 kb_repair(dry_run=true) 生成方案并向用户展示；
用户确认后用 kb_repair(dry_run=false) 应用并自动复审。
单点、简单的修改仍直接用 read + edit，不必走 kb_repair。
kb_repair 不会删除、移动或重命名笔记。
```

## 8. 测试计划

### `tests/test_tool_subagent.py`

- [ ] 子 Agent 能调用注入的工具并把结果回灌
- [ ] 未注册工具名被配对成错误结果而非抛出
- [ ] 超过 `max_steps` 返回 `max-steps` 并保留部分文本
- [ ] `cancelled` 置位后立即返回 `interrupted`

### `tests/test_kb_repair.py`

- [ ] `plan_tasks` 把同文件多条 `broken_link` 合并为一个任务
- [ ] `duplicate_title` 的多个 targets 进入同一任务
- [ ] `batch_tasks` 保证同批 `write_paths` 不相交
- [ ] `dry_run=true` 不修改任何文件
- [ ] 越界 `path`（`../`、kb_root 外）被拒绝
- [ ] `old_string` 出现 0 次或多次时跳过并记录
- [ ] 子 Agent 返回非 JSON 时任务标记 `failed`，其他任务照常完成
- [ ] 应用后自动复审，`audit_after` 的对应规则计数下降
- [ ] Bundle 安装后工具总数为 28 且包含 `kb_repair`
- [ ] `kb_repair_provider.dispose()` 后工具与提示词段落一并消失
- [ ] `kb_repair_enabled=False` 时不注册服务，其余工具不受影响

### 回归

- [ ] `tests/test_knowledge_base.py` 工具总数 27 → 28
- [ ] `tests/test_document_import.py` 工具总数同步
- [ ] 关闭知识库时工具数仍为 24

## 9. 验证命令

```powershell
cd F:\github\mini-harness

# 单元测试
uv run python -m unittest discover tests -v

# 编译检查
uv run python -m compileall -q src tests eval

# 工具清单（用假客户端，不消耗 API 额度）
uv run python -c "from mini_harness.bundle import BundleConfig, headless_bundle; from mini_harness.client import Message; from mini_harness.cordis import Context; C=type('C',(),{'MODEL':'fake','chat':lambda s,m,t=None: Message(role='assistant',content='x')}); ctx=Context(); ctx.plugin(headless_bundle, BundleConfig(settings_document={}, client=C())); print(sorted(ctx.require('agent').tool_names)); ctx.dispose()"

# JSON-RPC 配置冒烟
'{"jsonrpc":"2.0","id":1,"method":"settings.get","params":{"namespace":"agent"}}' | uv run mini-harness --rpc

# 端到端评测（消耗真实 API 额度）
uv run python eval/kb_governance_eval.py --case repair-concurrent --variant both --trials 1
```

端到端手工验收：在 `notes/` 放置 `index.md`（含 2 条断链）、`dup-a.md` / `dup-b.md`（同名标题）、`no-title.md`，然后依次输入：

```text
审计全库，列出问题，不要修改
用 kb_repair 生成修复方案，先不要应用
确认方案，应用并复审
```

预期：第二步 `dry_run=true` 时文件 mtime 不变；第三步后 `broken_link` 与 `duplicate_title` 计数归零。

## 10. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 子 Agent 输出非 JSON | 严格解析 + 任务级失败结算，不影响同批其他任务 |
| 子 Agent 越权改文件 | 只读工具投影 + `write_paths` 白名单 + kb_root 围栏三重校验 |
| 并发放大 API 成本 | `kb_repair_max_concurrency` 与 `max_tasks` 双上限；`dry_run` 默认 true |
| `old_string` 歧义 | 应用前唯一性检查，非唯一即跳过 |
| 应用期间文件被外部修改 | `filesystem.edit` 的 mtime 比对（`FS_STALE_VERSION`）兜底 |
| 修复引入新问题 | `kb_repair_auto_reaudit` 强制复审，以终态判定 |
| 线程不是安全边界 | 子 Agent 只读、无 shell，且不执行不可信代码 |

## 11. 回滚

单一开关：设置 `kb_repair_enabled=false`，或从 `bundle.py` 移除两行 `ctx.plugin(...)`。由于新增能力全部封装在 `kb_repair.py` / `tool_subagent.py` 中，且未改动 Agent 主循环，回滚不影响审计、导入、监听三条既有链路。

## 12. 工作量估计

| 阶段 | 内容 | 估计 |
| --- | --- | --- |
| 1 | `tool_subagent.py` + 测试 | 0.5 天 |
| 2–3 | 工具投影、任务规划、冲突分批 | 0.5 天 |
| 4–5 | 并发执行、ChangeSet 解析与应用 | 1 天 |
| 6–7 | 插件接线、配置、提示词 | 0.5 天 |
| 8 | 测试与回归 | 0.5 天 |
| 9 | 评测案例与 README | 0.5 天 |

合计约 3.5 天。

## 13. 面试讲解要点

> 审计是确定性的全库单次扫描，不需要模型并发；并发的价值在审计之后。主 Agent 按写集把 issue 聚合成任务并做冲突分批，子 Agent 在受限只读工具集下并行完成上下文调查、产出结构化 ChangeSet，主 Agent 再串行应用并统一复审。这样既拿到了并发的吞吐，又用"只读并发 + 串行写入 + 终态验收"守住了一致性和可验证性。

## 14. 执行记录

实现完成，`uv run python -m unittest discover tests` 共 61 个用例通过（1 个跳过，符号链接用例在无权限系统上自动跳过），`compileall` 通过，工具清单为 28 项。相对本方案的差异如下，全部是为了让功能真正可用或避免静默失败：

| 位置 | 方案原文 | 实际实现 | 原因 |
| --- | --- | --- | --- |
| `kb_repair.py` | `_project_tools` | 公开为 `project_tools`，并在投影前校验 `allowed` 与 `FORBIDDEN_SUBAGENT_TOOLS` 不相交 | 需要单测直接覆盖白名单与黑名单两条路径 |
| `KnowledgeRepairReport` | 只有 `tasks` 计数 | 增加 `changesets` 字段 | `dry_run=true` 必须让用户看到具体改什么，只给 `change_count` 无法评审 |
| `KnowledgeRepairReport` | 无截断信息 | 增加 `truncated`，超出 `max_tasks` 的任务以 `status="skipped"` 逐条回报 | 避免超过上限时静默丢弃待修问题 |
| `plan_tasks` | `(report, *, issue_ids, max_tasks)` | 增加 `rules`、`index_page`、`kb_prefix` | `orphan` 需要知道入口页；`read` 工具按 workspace 解析路径，而 `changes[].path` 按 kb_root 解析，指令里必须同时给出两种路径 |
| `plan_tasks` | 仅截断版本 | 另有 `plan_all_tasks` 返回未截断结果 | 服务层据此区分「没规划出任务」与「超过上限」 |
| 并发引擎 | 直接复用 `workflow` | `kb_repair_max_concurrency` 与 `jobs_max_concurrency` 相同时复用共享引擎，不同才另建 | 既满足"默认沿用"，又让配置真正生效 |
| `eval` | 新增 `--variant repair` | 另加 `--variant all` | 一次跑完 baseline / vertical / repair 三种组合 |
| 子 Agent 输出 | 要求纯 JSON | 解析前剥离 ```` ```json ```` 代码围栏 | 模型高频以代码块包裹 JSON，剥离后仍严格 `json.loads` |

`issue_ids` 中出现无法自动修复或不存在的 id 时直接抛错而不是静默忽略；`plan_tasks` 与 `apply_changeset` 的路径校验各自独立，越界路径在解析阶段就会让整个 ChangeSet 作废。
