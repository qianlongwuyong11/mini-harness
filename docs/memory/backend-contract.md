# P1-05：MemoryBackend 最小契约

本步定义 mini-harness 自己的调用方式。生产代码只有 `memory/contracts.py` 和包入口；FakeMemoryBackend 位于 tests，使用标准库，无新增依赖。HTTP 适配器在 P1-07/P1-08 实现。

## 阅读顺序

1. `MemoryScope`：由可信应用配置提供 user_id、project_id，两个字段共同决定隔离范围。后端负责映射远端身份，工具调用者不能自选项目身份。
2. `MemoryMessage`：只接受筛选后的 user/assistant 文本和来源消息 ID。筛选本身留给后续 capture 模块；此类型不自动检查密钥或不可信指令。
3. `MemoryBackend`：五个同步方法。网络方法可能阻塞，后续 worker 负责调度，不能直接放进同步 Session 事件回调。
4. `CommitResult`、`MemoryTask`：区分请求受理与后台完成。
5. `MemoryBackendError`：让调用方不依赖 httpx/OpenViking 的异常类型。

## 五个方法

| 方法 | 输入 | 输出/语义 |
|---|---|---|
| recall | scope、query、limit | MemoryHit 元组；无命中返回空元组，失败抛异常 |
| read | scope、命中中的 uri | MemoryDocument；必须校验 URI 归属 |
| append | scope、本地稳定 session_id、消息元组 | AppendResult；后端按需创建对应远端会话 |
| commit | scope、session_id | accepted + task_id，或没有待提交内容时 noop |
| get_task | scope、task_id | MemoryTask；必须验证任务属于该 scope |

URI 对上层是后端返回的不透明引用，上层无需理解 viking://。适配器仍需解析、规范化和限制 URI。召回结果保留 level，因为 P1-04 观察到结果可能包含摘要节点。score 可为空，也不承诺不同后端的分数可直接比较。

noop 是我们定义的业务语义，并非宣称已验证 OpenViking 空提交响应；HTTP 适配器必须根据真实响应映射，不能无依据返回成功。任务状态统一为 pending/running/completed/failed/cancelled；后续需要核对远端全部状态映射，未知状态不能当 completed。

## 错误与写入边界

错误码包括 invalid_input、not_found、permission_denied、unavailable、timeout、unknown_outcome、backend_error。message 要脱敏，不携带 SDK 对象和原始请求。

写入发生网络中断或批次部分成功时，如果不能确认远端是否生效，必须返回 UNKNOWN_OUTCOME。来源消息 ID 不保证去重；本接口没有自动重试。可靠同步、部分成功核对与崩溃恢复留待阶段三。

## FakeBackend 如何使用

Fake 只模拟协议：recall 用文本子串匹配；seed 预置记忆；commit 创建 pending 任务；complete 手动发布指定文档并把任务改为 completed。它不会把消息自动“提炼”为记忆，也没有持久化、线程安全、网络超时或故障恢复能力。

```python
scope = MemoryScope("alice", "aurora")
fake.append(scope, "session-1", (MemoryMessage("msg-1", "user", "Use SQLite"),))
receipt = fake.commit(scope, "session-1")
# 此时 get_task 为 pending，recall 还没有这条记忆。
fake.complete(scope, receipt.task_id, (MemoryDocument("fake://decision", "Use SQLite"),))
# 此时才可 recall/read；complete 是测试辅助方法，不属于 MemoryBackend。
```

两个单元测试覆盖受理/完成边界和不同用户/项目的范围限制，已通过。本步没有验证真实 HTTP Adapter 的隔离能力；P1-04 已确认当前 OpenViking dev 模式允许 ROOT 跨范围读取，因此后续 Adapter 必须实现约束。

复现：在项目根目录设置 `PYTHONPATH=src` 后，运行 `.venv/Scripts/python.exe -m unittest discover -s tests -p test_memory_contracts.py -v`。
