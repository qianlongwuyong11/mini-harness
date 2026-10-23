"""持续对话的 Agent 循环，以及供插件接入的生命周期事件。

Agent Loop 只固定 turn/step、消息配对和终止路径。Plan Mode、checkpoint、
retry、pruner 与 spill 都监听 Context 事件，因此增加或替换策略不需要修改
循环本身。
"""

from __future__ import annotations

import json
from time import perf_counter
from .telemetry import metric_scope
from .evaluation.trace import TraceStore, visible_message, wire_capture
from .retry import classify_failure
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from .client import ChatClient, Message, Tool, ToolCall
from .cordis import Context
from .inbox import Inbox
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .rpc import RpcDispatcher
from .session import Session


class EmptyResponseError(RuntimeError):
    code = "EMPTY_RESPONSE"


@dataclass
class StepBoundary:
    """`agent/pre-step` 的可变载荷；插件可贡献边界 narration。"""

    agent: Agent
    session: Session
    turn: int
    step: int
    notices: list[str] = field(default_factory=list)


@dataclass
class ModelRequest:
    """`agent/prepare-request` 与 `llm/request` 共享的请求 envelope。"""

    agent: Agent
    session: Session
    turn: int
    step: int
    system_prompt: str
    messages: list[Message]
    tools: list[Tool]
    schemas: list[dict[str, Any]]
    purpose: str = "normal"
    retry_of: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass
class ToolExecution:
    """工具管线载荷；post-execute 插件可以改写最终模型可见结果。"""

    agent: Agent
    session: Session
    call: ToolCall
    result: str = ""
    is_error: bool = False
    error_type: str | None = None
    error_code: str | None = None


class Agent:
    def __init__(
        self,
        ctx: Context,
        client: ChatClient,
        registry: ToolRegistry,
        assembler: PromptAssembler,
        *,
        variables: dict[str, str] | None = None,
        session: Session | None = None,
        session_id: str | None = None,
        trace_store: TraceStore | None = None,
        max_steps_per_turn: int = 10,
        trace_identity: dict[str, str] | None = None,
    ) -> None:
        if not 1 <= max_steps_per_turn <= 100:
            raise ValueError("max_steps_per_turn must be between 1 and 100")
        self._max_steps_per_turn = max_steps_per_turn
        self._evaluation_identity = dict(trace_identity or {})
        self._trace_store = trace_store or TraceStore(secrets=(getattr(client,"api_key",None),))
        self._last_model_request_id = None
        self._ctx = ctx
        self._client = client
        self._registry = registry
        self._assembler = assembler
        self._variables = variables
        self._inbox = Inbox()
        self._session = session or Session()
        self._turn_no = 0
        self._request_header: str | None = None
        self.id = f"agent-{uuid4().hex}"
        if session_id is not None and self._session.session_id is not None and session_id != self._session.session_id:
            raise ValueError("session_id conflicts with restored session")
        self.session_id = session_id or self._session.session_id or uuid4().hex
        self._session.session_id = self.session_id
        self.rpc_dispatcher: RpcDispatcher | None = None

    def followup(self, content: str) -> None:
        self._inbox.followup(Message(role="user", content=content))

    def steer(self, content: str) -> None:
        self._inbox.steer(Message(role="user", content=content))

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def session(self) -> Session:
        return self._session

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self._registry.all())

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._registry.schemas()

    def close(self) -> None:
        self._ctx.dispose()

    def run(self, max_turns: int = 5) -> Session:
        """同步处理 inbox；官方常驻驱动器会在空闲时挂起等待唤醒。"""
        while self._inbox.pending > 0 and self._turn_no < max_turns:
            claimed = self._inbox.claim_turn()
            if not claimed:
                break
            self._turn_no += 1
            turn_started = perf_counter()
            self._turn_event_id = self._session.append("turn/start", {"turn": self._turn_no, "metrics_version": 2}).id
            try:
                self._run_turn(claimed)
            except Exception as error:
                self._session.append(
                    "turn/end",
                    {"turn": self._turn_no, "reason": "error", "message": str(error), "elapsed_ms": (perf_counter()-turn_started)*1000},
                )
                raise
            else:
                self._session.append(
                    "turn/end", {"turn": self._turn_no, "reason": "completed", "elapsed_ms": (perf_counter()-turn_started)*1000}
                )
        return self._session

    def _run_turn(self, claimed: list[Message]) -> None:
        for step in range(1, self._max_steps_per_turn + 1):
            if step > 1:
                claimed = self._inbox.claim_step()

            boundary = StepBoundary(self, self._session, self._turn_no, step)
            self._ctx.emit("agent/pre-step", boundary)
            step_started = perf_counter()
            self._step_event_id = self._session.append("step/start", {
                "turn": self._turn_no, "step": step, "turn_event_id": self._turn_event_id,
                "metrics_version": 2,
            }).id
            step_error = None
            completed = False
            try:
                for message in claimed:
                    self._session.append("user/message", {"content": message.content, "source": "user_input"})
                for notice in boundary.notices:
                    self._session.append("user/message", {"content": notice, "source": "internal_notice"})

                request = self._assemble_request(step)
                self._ctx.emit("agent/prepare-request", request)
                self._record_request_header(request)
                reply = cast(
                    Message,
                    self._ctx.waterfall("llm/request", request, self._request_model),
                )
                self._append_assistant(reply)

                if not reply.tool_calls:
                    completed = True
                else:
                    for call in reply.tool_calls:
                        self._run_tool(call)
            except Exception as error:
                step_error = type(error).__name__
                raise
            finally:
                self._session.append("step/end", {
                    "turn": self._turn_no, "step": step,
                    "turn_event_id": self._turn_event_id, "step_event_id": self._step_event_id,
                    "elapsed_ms": (perf_counter() - step_started) * 1000,
                    "status": "error" if step_error else "completed",
                    "outcome": "error" if step_error else "answer" if completed else "tool_results",
                    "error_type": step_error,
                })

            if completed and not self._inbox.has_next_step:
                return
        raise RuntimeError(f"第 {self._turn_no} 轮超过 {self._max_steps_per_turn} 个 step 仍未结束")

    def _assemble_request(self, step: int) -> ModelRequest:
        system_prompt = self._assembler.render(self._variables)
        tools = self._registry.all()
        schemas = self._registry.schemas()
        return ModelRequest(
            agent=self,
            session=self._session,
            turn=self._turn_no,
            step=step,
            system_prompt=system_prompt,
            messages=[
                Message(role="system", content=system_prompt),
                *self._session.derive_messages(),
            ],
            tools=tools,
            schemas=schemas,
        )

    def _record_request_header(self, request: ModelRequest) -> None:
        header = {
            "config": {"provider": "deepseek", "model": self._client.MODEL},
            "system": request.system_prompt,
            "tools": request.schemas,
        }
        fingerprint = json.dumps(
            header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if fingerprint == self._request_header:
            return
        self._session.append(
            "request/header",
            {
                "header": header,
                "reason": "initial" if self._request_header is None else "change",
            },
        )
        self._request_header = fingerprint

    def _trace_identity(self):
        return dict(**self._evaluation_identity, session_id=self.session_id, agent_id=self.id,
            turn_event_id=getattr(self,'_turn_event_id',None),
            step_event_id=getattr(self,'_step_event_id',None))

    def _save_trace(self, value):
        try:
            return self._trace_store.put(value)
        except Exception as error:
            # Observability failure must not retry a successfully executed model/tool.
            self._session.append('trace/error',dict(**self._trace_identity(),
                error_type=type(error).__name__,model_request_id=self._last_model_request_id))
            return None

    def _request_model(self, request: ModelRequest) -> Message:
        request_id=uuid4().hex
        self._last_model_request_id=request_id
        identity=dict(**self._trace_identity(),model_request_id=request_id)
        input_ref=self._save_trace(dict(messages=[visible_message(m) for m in request.messages],
            tools=[dict(name=t.name,description=t.description,parameters=t.parameters) for t in request.tools],
            model=self._client.MODEL))
        wire_ref=None
        def capture(payload):
            nonlocal wire_ref
            wire_ref=self._save_trace(payload)
            self._session.append('trace/model-wire',dict(**identity,input_ref=wire_ref,
                reasoning_omitted=True,headers_recorded=False))
        self._session.append('trace/model-start',dict(**identity,step=request.step,
            purpose=request.purpose,retry_of=request.retry_of,input_ref=input_ref,
            input_kind='client_arguments',reasoning_omitted=True))
        if request.evidence_ids:
            self._session.append('rag/context',dict(**identity,evidence_ids=list(request.evidence_ids),
                input_ref=input_ref,purpose=request.purpose))
        started=perf_counter()
        reply=None
        status='error'
        error_type=None
        error_code=None
        failure=None
        try:
            with wire_capture(capture):
                reply = self._client.chat(request.messages, request.tools)
            if not reply.content and not reply.reasoning_content and not reply.tool_calls:
                raise EmptyResponseError('model returned a completed response with no content')
            status='ok'
            return reply
        except Exception as error:
            error_type=type(error).__name__
            failure=classify_failure(error)
            error_code=failure['code']
            raise
        finally:
            elapsed=(perf_counter()-started)*1000
            output_ref=self._save_trace(visible_message(reply)) if reply is not None else self._save_trace(dict(error=failure))
            self._session.append('metrics/model',dict(**identity,step=request.step,model=self._client.MODEL,
                purpose=request.purpose,retry_of=request.retry_of,input_ref=wire_ref or input_ref,
                input_kind='provider_payload' if wire_ref else 'client_arguments',output_ref=output_ref,
                provider_request_id=getattr(reply,'provider_request_id',None),
                elapsed_ms=elapsed,status=status,error_type=error_type,error_code=error_code,
                usage=reply.usage if reply is not None else None))

    def _append_assistant(self, reply: Message) -> None:
        self._session.append(
            "assistant/message",
            {
                **self._trace_identity(),
                "model_request_id": self._last_model_request_id,
                "content": reply.content,
                "source": "assistant_final" if not reply.tool_calls else "assistant_tool_call",
                **(
                    {"reasoning_content": reply.reasoning_content}
                    if reply.reasoning_content
                    else {}
                ),
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in reply.tool_calls
                ],
            },
        )

    def _run_tool(self, call: ToolCall) -> None:
        self._session.append(
            "tool/call",
            {**self._trace_identity(),"model_request_id":self._last_model_request_id,
             "call_id": call.id, "name": call.name, "arguments": call.arguments},
        )
        input_ref=self._save_trace(dict(name=call.name,arguments=call.arguments))
        execution = ToolExecution(self, self._session, call)
        started=perf_counter()
        error_type=None
        raw_output_ref=None
        pipeline_phase="pre_execute"
        phase_times={}
        try:
            with metric_scope(self._session, trace_store=self._trace_store, turn_event_id=getattr(self,'_turn_event_id',None),
                    step_event_id=getattr(self,'_step_event_id',None), call_id=call.id,
                    session_id=self.session_id,model_request_id=self._last_model_request_id):
                phase_started=perf_counter()
                self._ctx.emit("tools/pre-execute", execution)
                phase_times["pre_execute_ms"]=(perf_counter()-phase_started)*1000
                pipeline_phase="execute"
                phase_started=perf_counter()
                cast(ToolExecution,self._ctx.waterfall("tools/execute", execution, self._execute_tool))
                phase_times["execute_ms"]=(perf_counter()-phase_started)*1000
                raw_output_ref=self._save_trace(dict(content=execution.result,is_error=execution.is_error))
                pipeline_phase="post_execute"
                phase_started=perf_counter()
                self._ctx.emit("tools/post-execute", execution)
                phase_times["post_execute_ms"]=(perf_counter()-phase_started)*1000
                self._session.append("tool/result", {"call_id":call.id,"content":execution.result,"is_error":execution.is_error,
                    **self._trace_identity(),"model_request_id":self._last_model_request_id,
                    "error_type":execution.error_type,"error_code":execution.error_code})
                self._ctx.emit("tools/result", execution)
        except Exception as error:
            error_type=type(error).__name__
            raise
        finally:
            self._session.append('metrics/tool',dict(call_id=call.id,name=call.name,
                turn_event_id=getattr(self,'_turn_event_id',None),step_event_id=getattr(self,'_step_event_id',None),
                elapsed_ms=(perf_counter()-started)*1000,status='error' if error_type or execution.is_error else 'ok',
                session_id=self.session_id,model_request_id=self._last_model_request_id,
                phase_times=phase_times,pipeline_phase=pipeline_phase,raw_output_ref=raw_output_ref,
                input_ref=input_ref,output_ref=self._save_trace(dict(content=execution.result,is_error=execution.is_error)),
                error_code=execution.error_code or ('TOOL_PIPELINE_ERROR' if error_type else 'TOOL_ERROR' if execution.is_error else None),
                error_type=error_type or execution.error_type))

    def _execute_tool(self, execution: ToolExecution) -> ToolExecution:
        tool = self._registry.get(execution.call.name)
        if tool is None:
            execution.is_error = True
            execution.error_code = "UNKNOWN_TOOL"
            execution.error_type = "LookupError"
            execution.result = f"Error: 模型请求了未注册的工具 {execution.call.name!r}"
            return execution
        phase = "arguments"
        try:
            arguments = json.loads(execution.call.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("工具参数必须是 JSON 对象")
            phase = "execute"
            execution.result = tool.execute(arguments)
        except Exception as error:  # noqa: BLE001 - 工具边界必须把失败配对成 result
            execution.is_error = True
            execution.error_type = type(error).__name__
            execution.error_code = "INVALID_TOOL_ARGUMENTS" if phase == "arguments" else getattr(error,"code",None) or "TOOL_EXECUTION_ERROR"
            execution.result = f"工具执行出错: {error}"
        return execution
