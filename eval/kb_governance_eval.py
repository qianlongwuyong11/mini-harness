"""个人知识库治理 Agent 的最小 A/B 效果评测。

默认调用真实 DeepSeek 模型。fixture 在临时目录生成，评分只看文件终态、
治理规则复审和 Session，不相信模型在最终回答中的自我声明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini_harness.agent import Agent
from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.cordis import Context
from mini_harness.knowledge_base import KnowledgeBaseService
from mini_harness.persistence import JsonlStore

RESULTS_ROOT = Path.cwd() / ".mini-harness" / "evals" / "knowledge-base"


@dataclass(frozen=True)
class EvalCase:
    id: str
    task: str
    files: dict[str, str]
    rules: tuple[str, ...]
    allowed_changes: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    case: str
    variant: str
    trial: int
    passed: bool
    completed: bool
    remaining_issues: int
    unexpected_changes: tuple[str, ...]
    tool_calls: dict[str, int]
    tool_errors: int
    steps: int
    duration_ms: int
    session_path: str | None


CASES = (
    EvalCase(
        id="broken-link",
        task=(
            "治理这个知识库中的本地文档断链。找到正确目标并做最小修改，"
            "然后复审确认 broken_link 已消失；不要改动目标笔记。"
        ),
        files={
            "index.md": "# Index\n\n[生成器](notes/generators.md)\n",
            "notes/generator.md": "# Python 生成器\n\n生成器按需产生值。\n",
        },
        rules=("broken_link",),
        allowed_changes=("index.md",),
    ),
    EvalCase(
        id="duplicate-title",
        task=(
            "治理知识库中的重复标题。阅读两篇笔记，根据正文为其中一篇改成"
            "更准确且不重复的标题，然后复审 duplicate_title。"
        ),
        files={
            "index.md": "# Index\n\n[A](notes/a.md)\n[B](notes/b.md)\n",
            "notes/a.md": "# Python 基础\n\n介绍变量与表达式。\n",
            "notes/b.md": "# Python 基础\n\n介绍上下文管理器。\n",
        },
        rules=("duplicate_title",),
        allowed_changes=("notes/a.md", "notes/b.md"),
    ),
    EvalCase(
        id="orphan-note",
        task=(
            "治理知识库中的孤立笔记。把孤立笔记以合适的标题加入 index.md，"
            "不要改动笔记正文，然后复审 orphan。"
        ),
        files={
            "index.md": "# Index\n",
            "notes/context-manager.md": "# 上下文管理器\n\nwith 语句管理资源。\n",
        },
        rules=("orphan",),
        allowed_changes=("index.md",),
    ),
    EvalCase(
        id="clean-noop",
        task=(
            "审计这个知识库的标题、断链和重复标题。如果没有问题，不要修改"
            "任何文件，直接报告复审结果。"
        ),
        files={
            "index.md": "# Index\n\n[Python](notes/python.md)\n",
            "notes/python.md": "# Python\n\n一篇正常笔记。\n",
        },
        rules=("missing_title", "broken_link", "duplicate_title"),
        allowed_changes=(),
    ),
    EvalCase(
        id="combined",
        task=(
            "完整治理这个知识库：修复断链、重复标题和孤立笔记。先审计，"
            "只做必要修改，最后再次审计确认这三类问题均已消失。"
        ),
        files={
            "index.md": "# Index\n\n[错误目标](notes/missing.md)\n[A](notes/a.md)\n",
            "notes/a.md": "# 并发笔记\n\n介绍线程。\n",
            "notes/b.md": "# 并发笔记\n\n介绍 asyncio。\n",
            "notes/target.md": "# 正确目标\n\n链接应该指向这里。\n",
        },
        rules=("broken_link", "duplicate_title", "orphan"),
        allowed_changes=("index.md", "notes/a.md", "notes/b.md"),
    ),
)


def build_eval_agent(*, vertical: bool) -> tuple[Context, Agent]:
    ctx = Context()
    ctx.plugin(
        headless_bundle,
        BundleConfig(
            settings_document={
                "agent": {
                    "sandbox_mode": "workspace-write",
                    "shell_mode": "read-only",
                    "kb_stale_days": 180,
                    "kb_root": ".",
                    "kb_watch_enabled": False,
                }
            },
            enable_console_questions=False,
            enable_knowledge_base=vertical,
        ),
    )
    agent = ctx.require("agent")
    assert isinstance(agent, Agent)
    return ctx, agent


def run_case(case: EvalCase, variant: str, trial: int) -> RunResult:
    vertical = variant == "vertical"
    with tempfile.TemporaryDirectory(prefix=f"kb-eval-{case.id}-") as tmp:
        workspace = Path(tmp)
        for relative, content in case.files.items():
            _write(workspace / relative, content)
        before = _hashes(workspace)
        previous = Path.cwd()
        ctx: Context | None = None
        started = time.monotonic()
        session_path: str | None = None
        try:
            os.chdir(workspace)
            ctx, agent = build_eval_agent(vertical=vertical)
            agent.followup(case.task)
            session = agent.run(max_turns=10)
            completed = _completed(session.events)
            after = _hashes(workspace)
            changed = {
                path
                for path in set(before) | set(after)
                if before.get(path) != after.get(path)
            }
            unexpected = tuple(sorted(changed - set(case.allowed_changes)))
            report = _scorer(workspace).audit(case.rules, max_issues=1000)
            passed = completed and report.issue_count == 0 and not unexpected
            calls = Counter(
                str(event.data["name"])
                for event in session.events
                if event.type == "tool/call"
            )
            tool_errors = sum(
                1
                for event in session.events
                if event.type == "tool/result" and event.data.get("is_error") is True
            )
            steps = sum(1 for event in session.events if event.type == "step/start")
            if not passed:
                target = RESULTS_ROOT / (
                    f"{variant}-{case.id}-trial-{trial}-{int(time.time())}.jsonl"
                )
                JsonlStore(target).save(session)
                session_path = str(target)
            return RunResult(
                case=case.id,
                variant=variant,
                trial=trial,
                passed=passed,
                completed=completed,
                remaining_issues=report.issue_count,
                unexpected_changes=unexpected,
                tool_calls=dict(sorted(calls.items())),
                tool_errors=tool_errors,
                steps=steps,
                duration_ms=round((time.monotonic() - started) * 1000),
                session_path=session_path,
            )
        finally:
            if ctx is not None:
                ctx.dispose()
            os.chdir(previous)


def aggregate(results: list[RunResult], trials: int) -> list[dict[str, Any]]:
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        groups[result.variant].append(result)
    summaries: list[dict[str, Any]] = []
    case_count = len({result.case for result in results})
    for variant, rows in sorted(groups.items()):
        by_case: dict[str, list[RunResult]] = defaultdict(list)
        for row in rows:
            by_case[row.case].append(row)
        pass_power = sum(
            len(case_rows) == trials and all(row.passed for row in case_rows)
            for case_rows in by_case.values()
        )
        summaries.append(
            {
                "variant": variant,
                "runs": len(rows),
                "pass@1": sum(row.passed for row in rows) / len(rows),
                f"pass^{trials}": pass_power / case_count,
                "avg_duration_ms": round(
                    sum(row.duration_ms for row in rows) / len(rows)
                ),
                "avg_tool_calls": round(
                    sum(sum(row.tool_calls.values()) for row in rows) / len(rows), 2
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, choices=(1, 3), default=1)
    parser.add_argument(
        "--variant", choices=("baseline", "vertical", "both"), default="both"
    )
    parser.add_argument("--case", choices=tuple(case.id for case in CASES))
    parser.add_argument("--list", action="store_true", help="只列出案例，不调用模型")
    arguments = parser.parse_args()
    selected = [case for case in CASES if not arguments.case or case.id == arguments.case]
    if arguments.list:
        for case in selected:
            print(f"{case.id}: {case.task}")
        return

    variants = (
        ("baseline", "vertical")
        if arguments.variant == "both"
        else (arguments.variant,)
    )
    results: list[RunResult] = []
    for variant in variants:
        for case in selected:
            for trial in range(1, arguments.trials + 1):
                result = run_case(case, variant, trial)
                results.append(result)
                print(json.dumps(asdict(result), ensure_ascii=False))
    print(
        json.dumps(
            {"summary": aggregate(results, arguments.trials)},
            ensure_ascii=False,
            indent=2,
        )
    )


def _scorer(root: Path) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        root=root,
        include_globs=("**/*.md", "**/*.txt"),
        exclude_dirs=(".git", ".mini-harness", ".agents", "node_modules"),
        required_frontmatter=(),
        stale_days=180,
        index_pages=("README.md", "index.md", "INDEX.md"),
    )


def _completed(events: tuple[Any, ...]) -> bool:
    endings = [event for event in events if event.type == "turn/end"]
    return bool(endings and endings[-1].data.get("reason") == "completed")


def _hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative == ".mini-harness" or relative.startswith(".mini-harness/"):
                continue
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
