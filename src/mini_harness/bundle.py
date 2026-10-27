"""Python 版 headless Bundle：声明插件清单，不承载业务逻辑。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from .agent import Agent
from .calculator import calculator
from .capabilities import (
    delegation_consumer,
    delegation_provider,
    filesystem_consumer,
    filesystem_provider,
    goal_todo_consumer,
    goal_todo_provider,
    interaction_consumer,
    jobs_provider,
    plan_provider,
    questions_provider,
    rpc_provider,
    settings_provider,
    shell_consumer,
    shell_provider,
    skills_consumer,
    skills_provider,
    subagents_provider,
    web_consumer,
    web_provider,
    workflow_provider,
)
from .client import ChatClient, DeepSeekClient
from .cordis import Context, depends
from .document_import import document_import_consumer, document_import_provider
from .kb_repair import kb_repair_consumer, kb_repair_provider
from .knowledge_base import kb_consumer, kb_provider
from .knowledge_watch import knowledge_watch_provider
from .policies import (
    checkpoint_plugin,
    meter_provider,
    pruner_plugin,
    retry_plugin,
    spill_plugin,
)
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .session import Session


@dataclass(frozen=True)
class BundleConfig:
    settings_document: Mapping[str, Any]
    enable_console_questions: bool = True
    checkpoint_flush: Callable[[Session], None] | None = None
    spill_root: str = ".mini-harness/spills"
    client: ChatClient | None = None
    session: Session | None = None
    enable_knowledge_base: bool = True
    enable_rag: bool = False
    rag_catalog_path: str = ".mini-harness/rag/vector-catalog.sqlite3"
    rag_runtime: Any = None
    session_id: str | None = None
    memory_config: Any = None
    memory_sync_config: Any = None


def session_provider(ctx: Context, session: Session | None) -> None:
    ctx.provide("session", session or Session())


def prompt_provider(ctx: Context, _config: Any) -> None:
    prompt = PromptAssembler()
    ctx.provide("prompt", prompt)
    ctx.effect(
        lambda: prompt.section(
            "persona",
            "你是 {{name}}，一个本地插件化任务助手。先调查再修改，使用工具验证结果，"
            "长任务用 Goal/Todo，独立任务可委派给 Subagent。",
            order=0,
        )
    )
    ctx.effect(
        lambda: prompt.section("rules", "回答要简洁：先给结论，再给过程。", order=100)
    )


def tools_provider(ctx: Context, _config: Any) -> None:
    ctx.provide("tools", ToolRegistry())


@depends("tools")
def calculator_plugin(ctx: Context, _config: Any) -> None:
    tools = ctx.tools
    assert isinstance(tools, ToolRegistry)
    ctx.effect(lambda: tools.register(calculator))


def llm_provider(ctx: Context, client: ChatClient | None) -> None:
    ctx.provide("llm", client if client is not None else DeepSeekClient())


@depends("llm", "tools", "prompt", "session")
def agent_provider(ctx: Context, _config: Any) -> None:
    client = cast(ChatClient, ctx.llm)
    tools = ctx.tools
    prompt = ctx.prompt
    session = ctx.session
    assert isinstance(tools, ToolRegistry)
    assert isinstance(prompt, PromptAssembler)
    assert isinstance(session, Session)
    agent = Agent(
        ctx,
        client,
        tools,
        prompt,
        variables={"name": "小知"},
        session=session,
        session_id=_config,
    )
    ctx.provide("agent", agent)


def headless_bundle(ctx: Context, config: BundleConfig) -> None:
    """挂载插件树；顺序只表达 provider/wrapper 优先级。"""
    ctx.plugin(settings_provider, config.settings_document)
    ctx.plugin(session_provider, config.session)
    ctx.plugin(prompt_provider)
    ctx.plugin(tools_provider)
    ctx.plugin(llm_provider, config.client)
    ctx.plugin(meter_provider)
    ctx.plugin(calculator_plugin)

    # waterfall 外层先安装：retry 每次 next() 都会重新经过 checkpoint。
    # checkpoint 先于 Plan 插件监听 pre-step，保持“先落盘、再提交边界选择”。
    ctx.plugin(retry_plugin)
    ctx.plugin(checkpoint_plugin, config.checkpoint_flush)
    ctx.plugin(pruner_plugin)
    ctx.plugin(spill_plugin, config.spill_root)

    ctx.plugin(filesystem_provider)
    ctx.plugin(filesystem_consumer)
    if config.enable_knowledge_base:
        ctx.plugin(kb_provider)
        ctx.plugin(kb_consumer)
        ctx.plugin(document_import_provider)
        ctx.plugin(document_import_consumer)
        ctx.plugin(knowledge_watch_provider)
        # kb_repair 依赖 workflow，而 workflow_provider 在分支之后挂载；
        # Cordis 的 fiber 会在依赖就绪后自动加载，顺序不影响正确性。
        ctx.plugin(kb_repair_provider)
        ctx.plugin(kb_repair_consumer)
    if config.enable_rag:
        # RAG 是可选能力：默认关闭时不导入其模块，后续引入 Docling、
        # Elasticsearch、LlamaIndex 等依赖也不会影响普通 Agent 启动。
        from .rag.plugin import rag_plugin

        ctx.plugin(rag_plugin, config.rag_runtime or config.rag_catalog_path)
    ctx.plugin(shell_provider)
    if config.memory_config is not None:
        # 默认关闭时不加载记忆插件或供应商适配器。
        from .memory.plugin import memory_plugin

        ctx.plugin(memory_plugin, config.memory_config)
    if config.memory_sync_config is not None:
        from .memory.automatic import automatic_memory_plugin
        ctx.plugin(automatic_memory_plugin, config.memory_sync_config)
    if config.memory_config is not None or config.memory_sync_config is not None:
        from .memory.status import memory_status_plugin
        ctx.plugin(memory_status_plugin, (config.memory_config, config.memory_sync_config))
    ctx.plugin(shell_consumer)
    ctx.plugin(skills_provider)
    ctx.plugin(skills_consumer)
    ctx.plugin(goal_todo_provider)
    ctx.plugin(goal_todo_consumer)
    ctx.plugin(questions_provider, config.enable_console_questions)
    ctx.plugin(plan_provider)
    ctx.plugin(web_provider)
    ctx.plugin(web_consumer)
    ctx.plugin(subagents_provider)
    ctx.plugin(jobs_provider)
    ctx.plugin(workflow_provider)
    ctx.plugin(delegation_provider)

    # 这些 consumer 先等待 agent provider，展示服务后到自动启动。
    ctx.plugin(interaction_consumer)
    ctx.plugin(delegation_consumer)
    ctx.plugin(rpc_provider)
    ctx.plugin(agent_provider, config.session_id)
