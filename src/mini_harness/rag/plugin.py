"""Optional RAG providers. Heavy parsing/retrieval services arrive later."""
from __future__ import annotations

from ..cordis import Context
from .catalog import SQLitePaperCatalog


def rag_plugin(ctx: Context, config: str | None) -> None:
    catalog = SQLitePaperCatalog(config or ".mini-harness/rag/catalog.sqlite3")
    catalog.recover_expired_jobs()
    ctx.provide("paper_catalog", catalog)
