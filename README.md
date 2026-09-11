# 17｜个人知识库治理 Agent 的完整组装

> 预计时间：90 分钟 ｜ 前置：完成第 01–16 章 ｜ 命令行任务调用真实 DeepSeek 模型

前 16 章分别实现了模型调用、工具、会话、文件操作、任务管理和子智能体等能力。本章把它们接到同一个程序中，并通过可卸载的知识库插件把通用运行时落到“个人知识库治理”场景：根据自然语言查找本地笔记，审计标题、元数据和链接问题，安全修改后再次复审。

最终示例采用命令行运行方式，官方称为 headless。它不启动网页、桌面界面或 HTTP 服务：普通模式从命令行接收一项任务，完成后把最后一条模型回复写到标准输出；`--rpc` 模式则通过标准输入和标准输出逐行收发 JSON-RPC，供其他进程调用。

## 学习目标

完成本章后，你将能够：

- 用 `Context`、服务和 `Bundle` 组装一个插件化智能体；
- 区分程序内部的插件能力与模型可以调用的工具；
- 说明第 10–16 章的能力分别接入运行过程中的哪个位置；
- 跟踪保存、重试、结果裁剪和外部存储在一个步骤中的先后顺序；
- 区分一次性任务出口与持续 JSON-RPC 入口；
- 识别 Python 教学实现与官方命令行运行组件之间的差异。

## 17.1 运行入口

安装依赖后，可以直接运行一次任务：

```bash
uv run mini-harness "查看当前项目并总结入口"
```

也可以直接以模块方式运行：

```bash
uv run python -m mini_harness "查看当前项目并总结入口"
```

程序把进度信息写到标准错误，把最终回答写到标准输出。最后一条 `turn/end` 的 `reason` 为 `completed` 时，进程退出码是 0，否则是 1。没有提供任务时，程序先打印用法并以退出码 2 结束，不会提前读取 API Key。

### 交互模式（持续提问）

如果希望在同一会话里连续提问，而不是每次重新启动进程，使用 `--interactive` 或 `-i`：

```bash
uv run mini-harness --interactive
```

启动后会进入 REPL：程序复用同一个 Agent 和 Session，每输入一行问题就执行一轮，历史对话会保留在上下文中。输入 `/exit`、`quit`，或按 `Ctrl+D` / `Ctrl+C` 退出；退出前会把会话保存到 `.mini-harness/sessions/<uuid>.jsonl`。

也可以带着首条消息直接进入交互模式：

```bash
uv run mini-harness -i "先看一下当前项目结构"
```

恢复上一次保存的会话并继续提问：

```bash
uv run mini-harness -i --resume .mini-harness/sessions/<会话文件>.jsonl
```

指定新会话文件的保存路径：

```bash
uv run mini-harness -i --session .mini-harness/sessions/my-chat.jsonl
```

交互模式下，Agent 若调用 `ask_user_question`，也会从标准输入读取答案，因此终端里可能出现两种提示：REPL 的 `>` 和问答工具的 `>`。普通一次性任务模式不受影响。

JSON-RPC 使用另一条入口：

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"settings.get","params":{"namespace":"agent"}}' \
  | DEEPSEEK_API_KEY=... uv run mini-harness --rpc
```

`--rpc` 每读取一行请求就写出一行响应。目前提供 `settings.get`、`agent.run` 与 `plan.set` 三个方法。这只是第 16 章分发器的标准输入输出版本，不包含完整的宿主服务或网络 API 代理。

### 独立 JSON-RPC 命令行客户端

项目还提供一个单独的客户端命令。它启动 `mini-harness --rpc` 作为 Agent
子进程，客户端进程负责读取终端输入，并通过子进程的 stdin/stdout
发送和接收 JSON-RPC：

```bash
uv run mini-harness-client
```

也可以在启动时直接发送第一条任务：

```bash
uv run mini-harness-client "查看当前项目并总结入口"
```

客户端启动后可持续输入普通问题。同一个 Agent 子进程会一直存活，因此多次
`agent.run` 共享 Session 对话历史。输入 `/exit` 或按 `Ctrl+C` 关闭客户端；
客户端会先关闭子进程 stdin，让 Agent 保存 Session 后正常退出。

客户端还支持以下本地命令：

```text
/settings  查看 Agent 配置
/plan on   开启计划模式
/plan off  关闭计划模式
/help      显示帮助
/exit      保存会话并退出
```

这个客户端与 `--interactive` 的界面相似，但进程结构不同：

```text
mini-harness-client 进程
        │
        │ stdin/stdout 上的 JSON-RPC 2.0
        ▼
mini-harness --rpc Agent 子进程
```

`--interactive` 在一个进程内直接调用 `agent.followup()` 和 `agent.run()`；
`mini-harness-client` 则通过 `settings.get`、`plan.set` 和 `agent.run`
等 RPC 方法驱动另一个进程中的 Agent。

### 个人知识库治理

第一版面向本地文档目录。建议把 `F:\PersonalKB` 作为 workspace，再用
`--project` 指向本项目启动客户端：

```powershell
cd F:\PersonalKB
$env:DEEPSEEK_API_KEY="<你的 API Key>"
uv run --project F:\github\mini-harness mini-harness-client
```

Agent 启动后会创建并使用以下默认目录：

```text
F:\PersonalKB\
├─ inbox\       # 放入等待转换的原始文件
├─ notes\       # 标准 Markdown 知识库
├─ archive\     # 转换成功后保存原文件
└─ failed\      # 解析或写入失败的原文件
```

监听器每两秒扫描一次 `inbox`。文件大小和修改时间连续两次不变后才会处理，
避免读取仍在复制的文件；子目录结构会被保留，例如
`inbox\projects\design.docx` 会生成 `notes\projects\design.md`，原文件移动到
`archive\projects\design.docx`。转换后会立即运行一次不含 `stale` 的确定性
审计，并将导入状态和问题汇总输出到 Agent 进程的标准错误。

因为当前目录不是项目根目录，项目根目录的 `.env` 不会被自动发现；应像上面
一样设置环境变量，或者在 `F:\PersonalKB` 提供自己的 `.env`。监听器属于
Agent 进程的生命周期：`mini-harness-client` 退出后，监听线程也会停止。

可以直接使用自然语言：

```text
you> 找到我记录异步并发控制的笔记，并说明最相关的三篇
you> 把 external\架构设计.docx 导入为 架构设计.md，标签是 Python 和 Agent
you> 审计全部笔记的断链和重复标题，先给计划，不要直接修改
you> 修复已经确认的断链，完成后重新审计
```

知识库插件提供四个模型工具：

- `kb_search`：按标题、路径、frontmatter 和正文进行本地加权检索，返回候选
  路径、匹配位置与片段；这是关键词及中文片段匹配，不是 Embedding 语义检索。
- `kb_audit`：确定性检查 `missing_title`、`missing_frontmatter`、
  `broken_link`、`duplicate_title`、`orphan` 和 `stale`。这些规则由 Python
  判断，模型负责理解需求、选择规则、阅读上下文、决定修复和生成报告。
- `kb_import_document`：读取 workspace 内的 `.pdf`、`.docx`、`.md`、`.txt`、
  `.html`、`.htm`、`.json` 或 `.csv`，提取文本并写成知识库内的 UTF-8
  Markdown。参数包括必填的 `source_path`，以及可选的 `destination`、`title`
  和 `tags`。
- `kb_repair`：批量修复 `kb_audit` 发现的问题。可选参数是 `issue_ids`、
  `rules`、`dry_run`（默认 `true`）和 `max_tasks`。它不会删除、移动或重命名
  笔记。

导入结果会生成平面 frontmatter：

```markdown
---
title: "架构设计"
tags: "Python, Agent"
source: "archive/架构设计.docx"
updated: 2026-09-11
---

# 架构设计
```

`source_path` 可以是相对 workspace 的路径或 workspace 内的绝对路径；
`destination` 必须是相对 `kb_root`（默认 `notes`）的 `.md` 路径。手动工具
省略目标时写入 `<kb_root>/<kb_import_dir>/<源文件名>.md`，并保留原文件；
监听器则负责移动成功或失败的收件箱文件。两条路径都拒绝覆盖已有目标，并把
单个源文件限制在 `kb_import_max_bytes` 以内。老式 `.doc` 和扫描版 PDF 的
OCR 暂不支持；PDF 某些页面提取不到文本时会在结果中返回 warning。

治理闭环为：

```text
inbox 新文件 → 等待稳定 → 归档原文件 → kb_import_document → 标准 Markdown
                                              → kb_audit → 状态输出
自然语言任务 → kb_search/kb_audit → read/grep → 治理计划
             → edit/write → kb_audit 复审 → 最终报告
批量修复任务 → kb_audit → kb_repair(dry_run=true) → 用户确认
             → kb_repair(dry_run=false) → kb_audit 复审 → 最终报告
```

`kb_audit` 只读；单点修复继续使用原有文件工具，因此保留 workspace 围栏、
read-before-write 和 mtime 并发检查。监听后的审计是确定性检查，不会在后台
并发调用模型自动改写笔记；用户可以随后要求 Agent 查看并修复审计问题。

### 并发修复的边界

`kb_repair` 把「审计—调查—应用—复审」拆成四段，并发只出现在调查这一段：

1. 主 Agent 按写集（`write_paths`）把 issue 聚合成任务。同一文件的多条断链
   合并为一个任务；`duplicate_title` 涉及的全部笔记必须进同一个任务，否则两
   个子 Agent 各改各的，改完仍然重复；`orphan` 需要改入口页，写集里因此包含
   `index.md`。
2. 冲突分批保证同一批内任意两个任务的 `write_paths` 不相交，同批并发、批与
   批之间串行。共享入口页的任务会自然落到不同批次。
3. 子 Agent 只拿到从主注册表投影出来的只读工具子集
   （`read`、`grep`、`glob`、`kb_search`），产出结构化 `ChangeSet` JSON，
   不写任何文件，也不能再往下委派。
4. 主 Agent 串行应用 `ChangeSet`。每条编辑都要通过 kb_root 围栏、
   `write_paths` 白名单、`old_string` 唯一性三重校验，再交给
   `filesystem.edit`，由沙箱与 mtime 比对兜底。单条失败不回滚其他条目，但会
   如实记录在 `RepairTaskResult.error` 中。

`dry_run=true`（默认）只返回预览，不写盘；`dry_run=false` 应用后会按
`kb_repair_auto_reaudit` 自动复审，报告里同时给出 `audit_before` 与
`audit_after`，因此验收依据是文件终态而不是模型自述。子 Agent 输出非 JSON、
越权写入白名单外的路径，或 `old_string` 不唯一时，对应任务标记为 `failed`
或跳过，不影响同批其他任务。

可以先列出内置评测案例而不调用模型：

```bash
uv run python eval/kb_governance_eval.py --list
```

配置 API Key 后，用同一案例对比通用基线和知识库插件版：

```bash
uv run python eval/kb_governance_eval.py \
  --case broken-link --variant both --trials 1
```

`--variant repair` 会额外打开 `kb_repair`，用于对比「逐个 read + edit」和
「只读并发调查 + 串行应用」两条修复路径；`--variant all` 一次跑完三种组合：

```bash
uv run python eval/kb_governance_eval.py \
  --case repair-concurrent --variant all --trials 1
```

`--trials 3` 会调用更多次真实模型并产生相应费用，用于比较 pass^3 稳定性。
评测只根据修复后的文件终态、再次审计结果和 Session 判分，不采用模型自评。

## 17.2 一切皆插件

Cordis 内核本身不实现模型调用、文件读写或任务管理。它只负责安装插件、连接插件依赖、分发事件，以及在卸载时清理资源。模型、工具、会话、提示词、运行循环、文件系统、技能和任务调度都由插件提供或扩展。这就是“一切皆插件”：内核保持精简，具体能力通过明确的扩展位置组合起来。

插件不一定是模型能够调用的工具。`skill`、`ask_user_question` 和 `job_output` 会出现在模型的工具清单中；模型连接、会话、自动重试、关键节点保存和上下文裁剪等插件只在程序内部工作，不应出现在工具清单中。最终示例一共向模型注册 28 个工具，其中 `kb_search`、`kb_audit`、`kb_import_document` 和 `kb_repair` 来自知识库场景插件；其他插件负责支撑这些工具和运行循环。

`knowledge_watch_provider` 就是一个不注册模型工具的后台 Provider。它依赖
`document_import` 和文件沙箱服务，安装时启动监听线程，卸载时设置停止事件并
等待线程退出；因此没有把轮询逻辑塞入 Agent 主循环，也不会增加模型工具数量。

`kb_repair_provider` 展示了另一种复用方式：它不新建第二套工具体系，而是从
主 `ToolRegistry` 投影出只读子集交给子 Agent，并复用 `WorkflowEngine` 做并发、
复用 `FilesystemService` 做写入。因此工具定义、沙箱围栏和 read-before-write
都只有一份实现。`kb_repair_provider` 依赖 `workflow`，而 `workflow_provider`
在知识库分支之后才挂载；Cordis 的 fiber 会在依赖就绪后自动加载，安装顺序不
影响正确性。

一个能力通常分成三层。服务定义（Service Definition）说明调用方可以依赖哪些方法；服务提供者（Provider）给出可以替换的具体实现；服务使用者（Consumer）再把这项服务接到提示词、模型工具或运行事件上。

例如，`filesystem_provider` 提供文件操作服务，`filesystem_consumer` 把其中五项操作注册成模型工具，并加入相应的安全提示。卸载使用者后，这些工具说明和提示词会一起消失；替换服务提供者时，使用者会先停止，再使用新服务重新启动。

## 17.3 用 Context、Bundle 和 build_agent 完成组装

`build_agent()` 现在只做三件事：创建根 Context，挂载 Python Bundle，从服务表取得 Agent。

```python
ctx = Context()
ctx.plugin(
    headless_bundle,
    BundleConfig(
        settings_document=load_settings_document(),
        enable_console_questions=enable_console_questions,
        checkpoint_flush=checkpoint_flush,
    ),
)
return cast(Agent, ctx.require("agent"))
```

`headless_bundle` 是一份明确的 Python 插件清单。它先安装配置、会话、提示词、工具注册表和模型连接，再安装文件、命令、技能、目标、计划、网络搜索、子智能体、后台任务、工作流和运行策略，最后提供完整的 `Agent` 服务。依赖 `agent` 服务的用户问答、任务委派和 RPC 插件会在服务出现后自动启动。

`Agent` 的构造器不直接接收重试、持久化检查点、计划模式或上下文控制策略。运行循环只在“步骤开始前”“准备请求”“调用模型”“执行工具”和“工具返回后”等位置发布事件，相关插件监听自己关心的位置。以后新增策略时，不需要继续修改主循环。

注册工具、提示词片段和 RPC 方法时都会返回对应的取消函数。插件通过 `ctx.effect(...)` 统一登记这些函数；`agent.close()` 只需关闭根 `Context`，运行环境就会按相反顺序卸载整棵插件树。可继续子智能体、后台任务、工具、提示词、RPC 方法和服务注册都会沿这条路径清理。

## 17.4 第 10–16 章接在哪里

完整示例一共注册 24 个模型工具，包含 `calculator` 和下面 23 个第 10–16 章工具。

| 章节 | 工具或服务 | 怎样接入最终示例 |
|---|---|---|
| 10 | `read`、`write`、`edit`、`grep`、`glob` | 文件服务提供操作，使用者注册工具和安全提示 |
| 11 | `shell` | 命令服务提供执行能力，使用者注册工具；教学版没有内核沙箱 |
| 12 | `skill` | 技能目录提供内容，使用者注册按需加载工具和菜单 |
| 13 | `get_goal`、`create_goal`、`update_goal`、`todo_write` | 目标与任务清单共用当前会话保存状态 |
| 13 | `ask_user_question`、`exit_plan_mode` | 用户问答和计划模式在智能体服务就绪后接入 |
| 14 | `subagent`、`subagent_fork`、`send_message`、`interrupt_agent` | 子智能体服务负责委派、继续对话和中断 |
| 14 | `job_output`、`job_list`、`job_kill` | 后台任务按根智能体隔离 |
| 14 | `workflow` | 工作流限制并发数和总任务数 |
| 15 | `web_search`、`web_fetch` | 搜索与网页抓取分别注册成工具 |
| 16 | `Settings`、`RpcDispatcher` | 配置服务和 RPC 方法在智能体、计划服务就绪后接入 |

所有相关插件都启动后，模型工具清单共有 24 项。计划模式只改变 `plan:policy` 提示词片段；`exit_plan_mode` 始终保留在工具说明中，但在非计划模式下调用会返回错误。如果卸载某个注册工具的插件，它提供的工具会从下一次模型请求中消失。

## 17.5 一次步骤中发生了什么

运行循环负责推进步骤，各个插件在约定的位置完成自己的工作：

1. 开始新步骤前，先保存上一批事件，再让已经批准的计划模式切换生效。
2. 写入 `step/start`，接收本步骤需要处理的用户消息。
3. 重新组装系统提示词、消息和工具说明，并估算它们占用的上下文空间。
4. 压力达到 80% 时，裁剪历史中的大型工具结果，再重新生成模型消息。
5. 请求内容发生变化时记录新的 `request/header`，然后在成功保存事件后调用模型。临时错误会在当前步骤中等待并重试。
6. 模型请求工具时，先保存 `tool/call`，再执行工具。工具返回超大文本时，保存完整原文，只把预览写入 `tool/result`。
7. 当前步骤的消息和工具结果都处理完后，写入 `step/end`。

如果检查点保存失败，模型和工具都不会继续执行。这保证了“先记录操作意图，再执行动作”，但不能保证外部操作只发生一次：程序仍可能在远端操作成功、结果尚未保存时退出。

## 17.6 计划模式与用户问答

`plan.set` 在两个轮次之间可以立即切换计划模式。如果智能体正在执行一个尚未结束的轮次，新选择会先保持待生效状态，到下一步骤开始时再提交。

JSON-RPC 的 `plan.set` 要求 `active` 是真正的 JSON boolean。字符串 `"false"` 不会按 Python truthiness 误转成 true，而是返回 `INVALID_PARAMS`。

`exit_plan_mode` 要求计划以 `#` 一级标题开头，并通过 UserQuestionService 提交结构化评审。只有唯一答案项、唯一选择 `Approve`、且没有 custom 文本时才算批准。

批准结果返回后，日志中的计划模式仍会暂时保持开启。同一条模型回复中剩余的工具调用继续遵守计划模式，下一步骤开始时才写入 `plan/mode=false`。这样可以避免一批工具调用执行到一半时改变规则。

普通命令行模式会在标准错误中显示问题、计划和选项，再从标准输入读取编号、标签或自定义反馈。`--rpc` 模式不会启用这项终端问答，因为 JSON-RPC 和 `input()` 不能同时读取同一个标准输入，当前的最小 RPC 接口也没有单独的问答通道。只有当前任务的根智能体能够发起用户问答，子智能体不能阻塞等待终端输入。

## 17.7 重试、保存与大结果处理

`RetryPolicy` 默认只重试空响应、限流、服务端错误、超时和网络传输错误，最多五次。等待时间从 500 毫秒逐次增加到最多 10 秒，并加入少量随机偏移，避免多个请求同时再次访问服务。

服务端返回合理的 `Retry-After` 时，程序优先采用它给出的等待时间。每次等待前先写入并保存 `llm/retry`，等待结束后再写入 `llm/retry-started`。失败结果不会进入模型消息；同一步骤的重试复用已经组装好的请求，不会重复执行步骤开始前的策略。

`ToolResultPruner` 按字符数判断工具结果大小。`TokenMeter` 先估算系统提示词、当前消息和工具说明的总长度；只有上下文压力达到 80% 时，裁剪器才处理超过阈值的旧结果。模型随后只看到保留的开头、省略标记和结尾，完整原事件仍保存在日志中。

`SpillPolicy` 按 UTF-8 字节数判断刚刚产生的工具结果。过大的纯文本先由 `LocalSpillStore` 保存，模型只收到预算内的首尾预览、文件位置和读取提示。`read` 工具支持从指定行开始、每次最多读取 2000 行，因此模型可以分段取回完整内容。保存失败或当前结果不适合外存时，程序保留原文，不会把一次成功的工具调用改成失败。

第 09 章的模型摘要压缩仍作为独立示例运行。本章已经接入长度估算、旧结果裁剪和新结果外存，但尚未接入自动摘要，也没有处理模型服务返回的输入过长错误。

## 17.8 子智能体与后台任务

`subagent` 可以创建一次性或可继续对话的子智能体，并选择在前台等待或放到后台运行。`subagent_fork` 固定创建一次性子智能体，同样可以选择前台或后台；它使用父会话中已经完成的历史作为起点。

全新子智能体从空会话开始，只看到完整的任务说明。分支子智能体只复制父会话到最后一个完整 `turn/end` 为止，当前尚未结束的轮次不会进入子会话。

可继续子智能体保留自己的会话，并按先到先得顺序处理消息。后台创建时，首条任务进入队列后便返回 `child_id` 和接收确认；`send_message` 也只确认消息已经入队，不等待回答。`interrupt_agent` 用于请求中断当前轮次。其他子智能体或根智能体不能只凭编号访问不属于自己的会话。

一次性后台任务由 `LocalJobs` 管理。每个任务绑定所有者，并支持列举、读取、等待和取消。排队中、运行中和正在停止的任务都会占用并发名额；取消请求发出后，只有任务真正停止才会进入已取消状态。可继续子智能体不属于普通后台任务。教学版没有完成通知、子智能体主动报告和跨进程恢复，因此父智能体若要自动取得后台子智能体的最终回答，还需要增加查询或通知机制。

`WorkflowEngine` 使用 Python 线程运行任务函数，支持并行执行和分阶段执行，并限制并发数与总任务数。线程不是安全隔离，不能执行不可信代码。模型调用 `workflow` 时只需提供简化后的 `tasks[]` 参数。官方参考版本使用 Worker Thread 执行受限 JavaScript 脚本，功能和隔离方式更完整。

`run_subagent` 只做纯文本委派：它不给子智能体任何工具，因此子智能体无法核实事实。`run_tool_subagent` 补上这一层——同样是独立会话、不带父历史，但每一步都把调用方传入的工具集交给模型，并把工具结果以 `role="tool"` 回灌。工具集由调用方投影，子智能体不能自行扩权；未授权的工具名、非法参数和执行异常都被配对成工具结果，不会中断循环。`kb_repair` 就是它的第一个使用者：投影出来的四个工具全部只读，所以「线程不是安全边界」这一限制在这里是成立的——子智能体既不写盘也不执行不可信代码。

## 17.9 配置项 Settings

默认配置位于 `agent` 命名空间，合并顺序仍是程序默认值、当前组合的基础值、用户设置，后者覆盖前者。

用户文档默认读取 `.mini-harness/settings.json`，例如：

```json
{
  "agent": {
    "sandbox_mode": "workspace-write",
    "shell_mode": "read-only",
    "approval_policy": "never",
    "retry_max_retries": 3,
    "spill_max_inline_bytes": 8192,
    "jobs_max_concurrency": 4,
    "workflow_max_agents": 16,
    "kb_root": "notes",
    "kb_include_globs": ["**/*.md", "**/*.txt"],
    "kb_exclude_dirs": [".git", ".mini-harness", ".agents", "node_modules", "inbox", "archive", "failed"],
    "kb_required_frontmatter": ["title"],
    "kb_stale_days": 180,
    "kb_index_pages": ["README.md", "index.md", "INDEX.md"],
    "kb_import_dir": ".",
    "kb_import_max_bytes": 20971520,
    "kb_watch_enabled": true,
    "kb_watch_dir": "inbox",
    "kb_archive_dir": "archive",
    "kb_failed_dir": "failed",
    "kb_watch_interval_seconds": 2,
    "kb_watch_stable_scans": 2,
    "kb_repair_enabled": true,
    "kb_repair_max_concurrency": 4,
    "kb_repair_max_tasks": 16,
    "kb_repair_subagent_max_steps": 6,
    "kb_repair_auto_reaudit": true
  }
}
```

上述所有目录都必须位于当前 workspace 内。`kb_required_frontmatter` 默认是
空数组，上例将 `title` 设为必填仅用于演示。`kb_import_dir` 是手动导入工具
相对 `kb_root` 的默认目录；监听导入会按照 `inbox` 的相对路径直接写到
`kb_root`。`kb_import_max_bytes` 默认是 20 MiB。frontmatter 教学实现只解析
平面的 `key: value`。把 `kb_watch_enabled` 设为 `false` 可以保留手动导入工具
但不创建目录、不启动监听线程。

`kb_repair_max_concurrency` 是只读子 Agent 的并发上限，默认与
`jobs_max_concurrency` 相同，此时直接复用共享的 `WorkflowEngine`，只有配置不
同才会另建一个引擎。`kb_repair_max_tasks` 是单次调用规划的任务数上限，超出
部分会以 `skipped` 状态如实回报，而不是被悄悄丢弃。
`kb_repair_subagent_max_steps` 限制单个子 Agent 的步数，用于兜住不收敛的调
查。`kb_repair_auto_reaudit` 控制应用后是否强制复审。把 `kb_repair_enabled`
设为 `false` 时 provider 直接返回，不注册服务与工具，审计、导入、监听三条链
路不受影响。

知识库审计不会建立远端索引，但模型请求、Session JSONL 和 Spill 文件可能
包含私人笔记片段。DeepSeek 是远端 API，因此敏感内容仍应先脱敏，不能把
“没有向量数据库”误解为“笔记绝不离开本机”。

更新配置时会检查 `revision` 版本号，避免旧数据覆盖新修改。本章只在启动时读取一次配置，没有实现运行时自动重新加载、通过 RPC 保存配置或在多套配置之间动态切换。

## 17.10 会话文件与退出

普通任务开始时会创建唯一的 `.mini-harness/sessions/<uuid>.jsonl` 文件。运行器把同一个 `JsonlStore.save` 交给检查点插件，因此模型请求和工具调用意图会在运行过程中逐步写入磁盘。无论任务成功还是发生错误，关闭根 `Context` 前都会再保存一次，保证 `step/end` 和 `turn/end(error)` 不只停留在内存中。RPC 模式退出时也执行相同收尾。

超大工具结果默认保存在 `.mini-harness/spills/<session>/`。工具给出的建议名称只用于生成文件名，存储服务会清理不安全字符、避免重名覆盖，并返回绝对路径。

程序从会话中找到最后一条非空模型回复作为最终结果。完成状态只读取最后一条 `turn/end`，不能因为较早轮次成功，就把随后发生的错误误报为任务完成。

## 对照官方 rc.8

本章的官方源码对照版本是 tag `dsh-v0.1.0-rc.8`、commit `141eb6fef83422698aef7a981029e843e8161534`。

如果要继续研究完整架构，可以对照官方文档的 [Cordis Primer](https://deepseek-harness.github.io/deepseek-harness/reference/cordis-primer)、[Capability Seams](https://deepseek-harness.github.io/deepseek-harness/reference/capability-seams) 与 [Extension Cookbook](https://deepseek-harness.github.io/deepseek-harness/reference/cookbook/extension-cookbook)。它们分别介绍插件内核、服务定义与使用关系，以及新增能力应接入扩展位置而不是不断修改智能体主循环的原则。

| 官方源码 | 教学版对应 | 边界 |
|---|---|---|
| `packages/bundle/base/cordis.patch.yml`、`packages/bundle/headless/cordis.patch.yml` | `cordis.py`、`bundle.py`、`build_agent` | Python 版使用明确的插件清单，不实现 YAML 配置加载、热重载和隔离作用域 |
| `vendor/cordis/src/context.ts`、`fiber.ts`、`reflect.ts`、`events.ts` | `Context`、`PluginHandle`、`depends`、`waterfall` | 保留插件生命周期、等待依赖、替换服务和自动清理；同步教学版不实现异步插件任务 |
| `packages/plan/plan-mode`、`packages/interaction/user-questions` | `plan.py`、`user_questions.py` | 保留日志折叠、稳定工具、评审和边界提交 |
| `packages/session/session-checkpoint-policy` | `checkpoint.py` 与模型请求、工具执行、步骤开始和重试前的保存位置 | 前三个位置与官方一致；教学版没有后台批量保存，因此还会显式保存重试调度事件 |
| `packages/llm/llm-retry` | `retry.py` | 实现次数有限的重试；没有始终重试和异步取消信号 |
| `packages/compaction/compaction-tool-result-pruner` | `pruner.py` | 教学版只处理纯文本消息块 |
| `packages/spill/*` | `spill.py` | 使用本地存储服务和外存策略；不处理分发日志中的结果 |
| `packages/subagent/*`、`packages/jobs/jobs-local` | `subagent.py`、`jobs.py` | 使用当前进程中的线程；没有远程服务和跨进程恢复 |
| `packages/workflow/workflow-worker-thread` | `workflow.py` | 使用 Python 函数讲解工作流，线程不构成安全隔离 |
| `packages/api/gateway` | `rpc.py`、`--rpc` | 只实现最小 JSON-RPC 接口，不包含完整宿主服务和 API 代理 |

## 本章小结

- 第 10–16 章的能力已经通过服务和事件接入同一个完整示例；
- 智能体主循环不直接实现重试、保存、计划模式和上下文控制，`build_agent()` 只负责挂载插件集合；
- 计划模式与用户问答相互协作，但模式只在步骤边界切换；
- 重试、关键节点保存、结果裁剪和外部存储各自处理不同问题；
- 分支子智能体、可继续子智能体、后台任务和工作流分别处理历史继承、持续对话、后台执行与批量编排；
- 命令行入口支持一次性任务和标准输入输出上的 JSON-RPC，但不包含完整宿主服务或平台级沙箱。

## 练习

1. 选择一个真实任务，例如代码审查、资料研究或项目维护，画出它需要的插件树。区分哪些组件提供服务、哪些组件注册模型工具、哪些策略通过事件或 `waterfall` 处理链接入。
2. 假设要增加模型切换、产物导出或新的存储后端。说明这项能力应负责提供服务、使用服务还是监听运行事件，以及怎样做到卸载后不残留提示词、工具或监听器。
3. 为一次包含模型重试、大型工具结果和会话保存的步骤画出执行时间线。分析保存、重试、结果裁剪与外部存储的顺序如果改变，可能造成哪些数据丢失、重复调用或上下文浪费。
4. 基于本章完成一个端到端小项目：从命令行或 RPC 接收任务，使用至少两类工具，保存并恢复会话，最后输出可以核对的结果。记录实际启用的插件、关键事件和一个失败分支，不要求实现课程范围外的完整平台能力。
5. 当前教学版没有完整的宿主界面、平台级沙箱、SQLite、多模态、热重载和多智能体团队。请选择其中一项作为进一步完善的优先方向，说明它解决的真实问题、应接入哪个扩展位置，以及为什么其他能力可以暂缓。
