# 阶段 1：实际执行轨迹

2026-10-13 完成。普通 Agent 即可产生增强轨迹；未接入评测任务运行器、自动语义评分或新评测 UI。

## 新增记录

- 每次实际模型调用生成 model_request_id；metrics/model 保存 session_id、turn_event_id、step_event_id、purpose、retry_of 和输入输出文件引用。重试与引用修复都有独立请求 ID。
- trace/model-start 在调用前写入，异常或进程中断时仍能识别开始过的请求。trace/model-wire 在 DeepSeekClient 发送前记录真正的 JSON payload，包含裁剪和 RAG 注入后的最终 messages、工具 schema、模型参数；不保存认证头。
- 非 DeepSeek 自定义客户端只记录 client_arguments，明确区别于实际 provider payload；API 返回响应 id 时记录 provider_request_id。
- tool/call、tool/result、metrics/tool 关联发起调用的模型请求；工具原始结果和 post-execute 后模型可见结果分别保存，记录 pre_execute/execute/post_execute 的耗时。
- 工具异常保留 error_type 和 error_code：UNKNOWN_TOOL、INVALID_TOOL_ARGUMENTS、TOOL_EXECUTION_ERROR；RAG 范围错误有 NO_PAPER_SELECTED/PAPER_OUT_OF_SCOPE/PAPER_NOT_READY。范围校验和证据拒答行为未在本阶段改变。
- llm/retry、llm/retry-started、rag/repair 带事件关联；恢复 Session 后的重试计数以 step 事件身份区分，不再只按可重置的 turn 数字查旧记录。

## 文件与安全边界

evaluation/trace.py 不依赖 Pydantic/DeepEval；默认产物位于启动工作目录下 `.mini-harness/traces/<摘要前两位>/<sha256>.json`。Agent 构造器可注入 TraceStore 指定根目录。JSONL 仅保存相对路径、摘要、大小；阅读历史轨迹需要同时保留这一目录。

文件先临时写入并 fsync，再原子替换，成功后记录引用。相同脱敏内容复用同一摘要路径。文件写入异常记录 trace/error 和空引用；不会因此重新执行成功的模型/工具。该机制不等同于已有 Session JSONL 的崩溃持久性保证，Session 仍遵循原 checkpoint 策略。

快照使用独立脱敏副本，已知运行环境密钥、客户端密钥、结构化凭据字段、常见 Bearer/赋值字符串会隐藏；推理字段省略。常规正文和论文证据保留在本地。它不是任意敏感自然语言识别器，历史 Session 消息没有被回写脱敏。

阶段 1 没有添加读取任意磁盘文件的 HTTP 接口；未来评测 UI 必须在限定根目录内按摘要验证读取。没有真实模型外发。

## 计时与限制

model elapsed_ms 在最终调用边界计时，包含客户端序列化、请求内的 wire 快照写入、网络和响应解析，不是纯服务端推理时间；输出快照写入在计时完成后。Step 墙钟时间包含这些记录开销。工具整体耗时包含执行及中间快照写入，分阶段时间不包括阶段间文件写入，因此不要求各阶段之和等于整体。

当前没有首 token 时延，也未新增异步工具队列，不能伪造 queue latency。模型错误 usage 仍为空；原始工具输出和 post-execute 输出相同会复用文件。历史会话不补造缺失快照。评测 run/case/attempt 的外层映射留待阶段 3 运行器。

## 验证

新增 4 项集成/存储测试：内容寻址与脱敏；实际 HTTP payload 捕获及 provider ID；真实 loop 中超时重试、工具失败、最终注入上下文的关联；磁盘错误不触发重复模型调用。

全量回归 175 项：174 通过、1 跳过。使用本地 fixture 和模拟客户端，未调用远程聊天模型。新快照事件不投影到模型对话，现有 UI step 统计保持可用。

重启原 Python UI 服务后，新对话才会使用这版埋点。当前未自动重启正在运行的会话服务。
