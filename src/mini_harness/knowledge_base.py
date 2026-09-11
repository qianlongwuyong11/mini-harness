"""本地 Markdown/文本个人知识库的确定性审计插件。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .client import Tool
from .cordis import Context, depends
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .settings import SettingsScope

AUDIT_RULES = frozenset(
    {
        "missing_title",
        "missing_frontmatter",
        "broken_link",
        "duplicate_title",
        "orphan",
        "stale",
    }
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*]\(([^)]+)\)")
WIKI_LINK = re.compile(r"\[\[([^]|#]+)(?:#[^]|]*)?(?:\|[^]]*)?]]")
H1 = re.compile(r"^\s*#\s+(.+?)\s*$")
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:")
SEARCH_WORD = re.compile(r"[a-zA-Z0-9_+-]+|[\u4e00-\u9fff]+")
SEARCH_STOPWORDS = frozenset(
    {"我", "的", "了", "和", "与", "是", "在", "一下", "找到", "查找", "笔记", "内容"}
)


@dataclass(frozen=True)
class KbDocument:
    path: str
    title: str | None
    frontmatter: Mapping[str, str]
    modified_at: str
    links: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class GovernanceIssue:
    id: str
    rule: str
    severity: str
    path: str
    message: str
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class KbAuditReport:
    root: str
    document_count: int
    issue_count: int
    summary: Mapping[str, int]
    issues: tuple[GovernanceIssue, ...]
    truncated: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class KbSearchHit:
    path: str
    title: str | None
    score: int
    matched_in: tuple[str, ...]
    snippet: str


class KnowledgeBaseService:
    """扫描知识库并执行可重复、可测试的结构规则。"""

    def __init__(
        self,
        *,
        root: Path,
        include_globs: Sequence[str],
        exclude_dirs: Sequence[str],
        required_frontmatter: Sequence[str],
        stale_days: int,
        index_pages: Sequence[str],
        reference_date: date | None = None,
    ) -> None:
        self.root = root.resolve()
        self.include_globs = tuple(include_globs)
        self.exclude_dirs = frozenset(exclude_dirs)
        self.required_frontmatter = tuple(required_frontmatter)
        self.stale_days = stale_days
        self.index_pages = frozenset(_normal_path(item) for item in index_pages)
        self.reference_date = reference_date

    def audit(
        self, rules: Sequence[str] | None = None, *, max_issues: int = 200
    ) -> KbAuditReport:
        selected = set(rules or AUDIT_RULES)
        unknown = selected - AUDIT_RULES
        if unknown:
            raise ValueError(f"未知知识库规则: {', '.join(sorted(unknown))}")
        if not 1 <= max_issues <= 1000:
            raise ValueError("max_issues 必须在 1..1000 范围内")

        documents = self._scan()
        by_path = {document.path: document for document in documents}
        by_stem: dict[str, list[str]] = defaultdict(list)
        by_title: dict[str, list[str]] = defaultdict(list)
        for document in documents:
            by_stem[_normalize_title(Path(document.path).stem)].append(document.path)
            if document.title:
                by_title[_normalize_title(document.title)].append(document.path)

        issues: list[GovernanceIssue] = []
        incoming: Counter[str] = Counter()
        for document in documents:
            if "missing_title" in selected and not document.title:
                issues.append(
                    _issue(
                        "missing_title",
                        "error",
                        document.path,
                        "文档缺少 frontmatter title 或一级标题",
                    )
                )
            if "missing_frontmatter" in selected and document.path.endswith(".md"):
                missing = tuple(
                    key
                    for key in self.required_frontmatter
                    if not document.frontmatter.get(key, "").strip()
                )
                if missing:
                    issues.append(
                        _issue(
                            "missing_frontmatter",
                            "error",
                            document.path,
                            f"缺少必填 frontmatter: {', '.join(missing)}",
                            missing,
                        )
                    )

            for raw_link in document.links:
                resolved = self._resolve_link(
                    document.path, raw_link, by_path, by_stem, by_title
                )
                if resolved is not None:
                    incoming[resolved] += 1
                elif "broken_link" in selected:
                    issues.append(
                        _issue(
                            "broken_link",
                            "error",
                            document.path,
                            f"本地文档链接目标不存在: {raw_link}",
                            (raw_link,),
                        )
                    )

        if "duplicate_title" in selected:
            for normalized, paths in sorted(by_title.items()):
                if normalized and len(paths) > 1:
                    ordered = tuple(sorted(paths))
                    issues.append(
                        _issue(
                            "duplicate_title",
                            "error",
                            ordered[0],
                            "多个文档使用相同标题",
                            ordered,
                        )
                    )

        if "orphan" in selected and len(documents) > 1:
            for document in documents:
                if (
                    incoming[document.path] == 0
                    and _normal_path(document.path) not in self.index_pages
                ):
                    issues.append(
                        _issue(
                            "orphan",
                            "warning",
                            document.path,
                            "文档没有被知识库中的其他文档引用",
                        )
                    )

        if "stale" in selected:
            today = self.reference_date or date.today()
            for document in documents:
                updated = _document_date(document)
                if updated is not None and (today - updated).days > self.stale_days:
                    issues.append(
                        _issue(
                            "stale",
                            "warning",
                            document.path,
                            f"文档已超过 {self.stale_days} 天未更新",
                        )
                    )

        ordered_issues = sorted(
            issues, key=lambda item: (item.rule, item.path, item.targets, item.id)
        )
        summary = dict(sorted(Counter(item.rule for item in ordered_issues).items()))
        visible = tuple(ordered_issues[:max_issues])
        return KbAuditReport(
            root=str(self.root),
            document_count=len(documents),
            issue_count=len(ordered_issues),
            summary=summary,
            issues=visible,
            truncated=len(visible) < len(ordered_issues),
        )

    def search(self, query: str, *, limit: int = 10) -> tuple[KbSearchHit, ...]:
        if not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not 1 <= limit <= 20:
            raise ValueError("limit 必须在 1..20 范围内")
        normalized_query = _normalize_search(query)
        terms = _search_terms(query)
        hits: list[KbSearchHit] = []
        for document in self._scan():
            fields = {
                "title": _normalize_search(document.title or ""),
                "path": _normalize_search(document.path),
                "frontmatter": _normalize_search(
                    " ".join(
                        f"{key} {value}"
                        for key, value in document.frontmatter.items()
                    )
                ),
                "content": _normalize_search(document.content),
            }
            weights = {"title": 12, "path": 8, "frontmatter": 6, "content": 2}
            score = 0
            matched: set[str] = set()
            for name, text in fields.items():
                if normalized_query and normalized_query in text:
                    score += weights[name] * 4
                    matched.add(name)
                term_hits = sum(1 for term in terms if term in text)
                if term_hits:
                    score += weights[name] * term_hits
                    matched.add(name)
            if score:
                hits.append(
                    KbSearchHit(
                        path=document.path,
                        title=document.title,
                        score=score,
                        matched_in=tuple(sorted(matched)),
                        snippet=_matching_snippet(document.content, terms),
                    )
                )
        hits.sort(key=lambda item: (-item.score, item.path.casefold()))
        return tuple(hits[:limit])

    def _scan(self) -> tuple[KbDocument, ...]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"知识库目录不存在: {self.root}")
        paths: set[Path] = set()
        for pattern in self.include_globs:
            for path in self.root.glob(pattern):
                if not path.is_file() or self._excluded(path):
                    continue
                try:
                    path.resolve().relative_to(self.root)
                except ValueError:
                    continue
                paths.add(path)

        documents: list[KbDocument] = []
        for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            frontmatter, body = _frontmatter(text) if path.suffix == ".md" else ({}, text)
            title = _title_of(path, body, frontmatter)
            links = (
                _extract_links(text)
                if path.suffix.casefold() in {".md", ".txt"}
                else ()
            )
            modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            documents.append(
                KbDocument(
                    path=path.relative_to(self.root).as_posix(),
                    title=title,
                    frontmatter=frontmatter,
                    modified_at=modified.isoformat(),
                    links=links,
                    content=text,
                )
            )
        return tuple(documents)

    def _excluded(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return True
        return any(part in self.exclude_dirs for part in relative.parts[:-1])

    def _resolve_link(
        self,
        source: str,
        raw_link: str,
        by_path: Mapping[str, KbDocument],
        by_stem: Mapping[str, list[str]],
        by_title: Mapping[str, list[str]],
    ) -> str | None:
        if raw_link.startswith("wiki:"):
            target = raw_link.removeprefix("wiki:").strip()
            normalized = _normalize_title(Path(target).name)
            candidates = set(by_title.get(normalized, ()))
            candidates.update(by_stem.get(normalized, ()))
            if "/" in target or "\\" in target:
                candidates.update(
                    self._path_candidates(source, target, by_path, wiki=True)
                )
            return sorted(candidates)[0] if candidates else None
        target = raw_link.removeprefix("path:").strip()
        candidates = self._path_candidates(source, target, by_path, wiki=False)
        return sorted(candidates)[0] if candidates else None

    def _path_candidates(
        self,
        source: str,
        target: str,
        by_path: Mapping[str, KbDocument],
        *,
        wiki: bool,
    ) -> set[str]:
        clean = target.split("#", 1)[0].strip().strip("<>")
        if not clean or clean.casefold().startswith(EXTERNAL_SCHEMES):
            return set()
        suffix = Path(clean).suffix.casefold()
        if suffix and suffix not in {".md", ".txt"}:
            return set()
        base = self.root if wiki else self.root / Path(source).parent
        raw_path = base / clean
        variants = [raw_path]
        if not suffix:
            variants.extend((raw_path.with_suffix(".md"), raw_path.with_suffix(".txt")))
        resolved: set[str] = set()
        for candidate in variants:
            try:
                relative = candidate.resolve().relative_to(self.root).as_posix()
            except ValueError:
                continue
            if relative in by_path:
                resolved.add(relative)
        return resolved


@depends("filesystem", "agent_settings")
def kb_provider(ctx: Context, _config: Any) -> None:
    filesystem = ctx.filesystem
    settings = ctx.agent_settings
    assert isinstance(settings, SettingsScope)
    workspace = Path(filesystem.workspace).resolve()
    config = settings.get()
    configured_root = Path(str(config["kb_root"]))
    root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (workspace / configured_root).resolve()
    )
    try:
        root.relative_to(workspace)
    except ValueError as error:
        raise ValueError("kb_root 必须位于当前 workspace 内") from error
    service = KnowledgeBaseService(
        root=root,
        include_globs=_string_sequence(config["kb_include_globs"], "kb_include_globs"),
        exclude_dirs=_string_sequence(config["kb_exclude_dirs"], "kb_exclude_dirs"),
        required_frontmatter=_string_sequence(
            config["kb_required_frontmatter"], "kb_required_frontmatter"
        ),
        stale_days=_positive_int(config["kb_stale_days"], "kb_stale_days"),
        index_pages=_string_sequence(config["kb_index_pages"], "kb_index_pages"),
    )
    ctx.provide("knowledge_base", service)


@depends("knowledge_base", "tools", "prompt")
def kb_consumer(ctx: Context, _config: Any) -> None:
    service = ctx.knowledge_base
    tools = ctx.tools
    prompt = ctx.prompt
    assert isinstance(service, KnowledgeBaseService)
    assert isinstance(tools, ToolRegistry)
    assert isinstance(prompt, PromptAssembler)
    ctx.effect(
        lambda: prompt.section(
            "knowledge-base:governance",
            (
                "你当前的垂直角色是个人知识库治理 Agent。用户按自然语言查找笔记时"
                "先用 kb_search 召回候选，再用 read 核实；治理时先用 kb_audit 确定"
                "事实，再按 error 优先、warning 其次解释问题。修改前先 read，优先"
                "用 edit 做最小修改；不得自动删除、移动或重命名笔记，也不得为统一"
                "格式覆盖正文。批量修改前先给计划，修改后再次调用 kb_audit。最终"
                "报告需列出扫描数量、发现问题、修改文件、复审结果和未处理风险。"
            ),
            order=10,
        )
    )
    ctx.effect(
        lambda: tools.register(
            Tool(
                "kb_audit",
                (
                    "确定性审计本地个人知识库，检查缺少标题/元数据、断链、"
                    "重复标题、孤立和陈旧笔记；修改前后均应调用。"
                ),
                {
                    "type": "object",
                    "properties": {
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": sorted(AUDIT_RULES),
                            },
                            "uniqueItems": True,
                        },
                        "max_issues": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                },
                lambda arguments: _audit_tool(service, arguments),
            )
        )
    )
    ctx.effect(
        lambda: tools.register(
            Tool(
                "kb_search",
                (
                    "根据自然语言描述在本地知识库的标题、路径、frontmatter 和"
                    "正文中加权检索候选笔记；返回结果后用 read 核实内容。"
                ),
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
                lambda arguments: _search_tool(service, arguments),
            )
        )
    )


def _audit_tool(
    service: KnowledgeBaseService, arguments: Mapping[str, Any]
) -> str:
    raw_rules = arguments.get("rules")
    rules: tuple[str, ...] | None = None
    if raw_rules is not None:
        if not isinstance(raw_rules, list) or not all(
            isinstance(item, str) for item in raw_rules
        ):
            raise TypeError("rules 必须是字符串数组")
        rules = tuple(raw_rules)
    max_issues = arguments.get("max_issues", 200)
    if not isinstance(max_issues, int) or isinstance(max_issues, bool):
        raise TypeError("max_issues 必须是整数")
    return service.audit(rules, max_issues=max_issues).to_json()


def _search_tool(
    service: KnowledgeBaseService, arguments: Mapping[str, Any]
) -> str:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    limit = arguments.get("limit", 10)
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit 必须是整数")
    hits = service.search(query, limit=limit)
    return json.dumps(
        {
            "query": query,
            "count": len(hits),
            "hits": [asdict(hit) for hit in hits],
        },
        ensure_ascii=False,
        allow_nan=False,
    )


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return fields, "\n".join(lines[index + 1 :])
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    return {}, text


def _title_of(
    path: Path, body: str, frontmatter: Mapping[str, str]
) -> str | None:
    configured = frontmatter.get("title", "").strip()
    if configured:
        return configured
    if path.suffix.casefold() == ".txt":
        return next((line.strip() for line in body.splitlines() if line.strip()), None)
    for line in body.splitlines():
        match = H1.match(line)
        if match:
            return match.group(1).strip()
    return None


def _extract_links(text: str) -> tuple[str, ...]:
    links: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        raw = match.group(1).strip()
        target = raw.split(maxsplit=1)[0].strip("<>")
        if (
            target
            and not target.startswith("#")
            and not target.casefold().startswith(EXTERNAL_SCHEMES)
        ):
            suffix = Path(target.split("#", 1)[0]).suffix.casefold()
            if not suffix or suffix in {".md", ".txt"}:
                links.append(f"path:{target}")
    for match in WIKI_LINK.finditer(text):
        target = match.group(1).strip()
        if target:
            links.append(f"wiki:{target}")
    return tuple(dict.fromkeys(links))


def _document_date(document: KbDocument) -> date | None:
    for key in ("updated", "date"):
        raw = document.frontmatter.get(key)
        if raw:
            try:
                return date.fromisoformat(raw[:10])
            except ValueError:
                pass
    try:
        return datetime.fromisoformat(document.modified_at).date()
    except ValueError:
        return None


def _issue(
    rule: str,
    severity: str,
    path: str,
    message: str,
    targets: tuple[str, ...] = (),
) -> GovernanceIssue:
    fingerprint = json.dumps(
        [rule, path, list(targets)], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return GovernanceIssue(
        id=hashlib.sha256(fingerprint).hexdigest()[:16],
        rule=rule,
        severity=severity,
        path=path,
        message=message,
        targets=targets,
    )


def _normalize_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def _normalize_search(value: str) -> str:
    return " ".join(value.casefold().split())


def _search_terms(query: str) -> tuple[str, ...]:
    terms: set[str] = set()
    for word in SEARCH_WORD.findall(query.casefold()):
        if word in SEARCH_STOPWORDS:
            continue
        terms.add(word)
        if any("\u4e00" <= char <= "\u9fff" for char in word) and len(word) > 2:
            terms.update(word[index : index + 2] for index in range(len(word) - 1))
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


def _matching_snippet(content: str, terms: Sequence[str]) -> str:
    for line in content.splitlines():
        normalized = _normalize_search(line)
        if any(term in normalized for term in terms):
            compact = line.strip()
            return compact[:197] + "..." if len(compact) > 200 else compact
    compact = next((line.strip() for line in content.splitlines() if line.strip()), "")
    return compact[:197] + "..." if len(compact) > 200 else compact


def _normal_path(value: str) -> str:
    return Path(value).as_posix().casefold()


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{name} 必须是非空字符串数组")
    return tuple(value)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{name} 必须是正整数")
    return value
