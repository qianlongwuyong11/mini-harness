"""Headless paper Agent using the existing ChatClient, Session and checkpoint flow."""
import argparse
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mini_harness.bundle import BundleConfig,headless_bundle
from mini_harness.capabilities import load_settings_document
from mini_harness.cordis import Context
from mini_harness.persistence import JsonlStore
from mini_harness.rag.runtime import PaperRuntime,RagConfig


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('question')
    cli.add_argument('--catalog',default=str(ROOT/'.mini-harness/rag/vector-catalog.sqlite3'))
    cli.add_argument('--index',default=None)
    cli.add_argument('--resume',type=Path)
    args = cli.parse_args()
    runtime = PaperRuntime(RagConfig(catalog_path=args.catalog,index_name=args.index))
    store = JsonlStore(args.resume or ROOT/'.mini-harness/sessions'/f'{uuid4().hex}.jsonl')
    ctx = Context()
    try:
        session = store.load() if args.resume else None
        ctx.plugin(headless_bundle,BundleConfig(settings_document=load_settings_document(),
            enable_rag=True,rag_runtime=runtime,session=session,checkpoint_flush=store.save,
            enable_knowledge_base=False,enable_console_questions=False))
        agent = ctx.require('agent')
        agent.followup(args.question)
        try:
            agent.run()
        finally:
            store.save(agent.session)
        replies = [m.content for m in agent.session.derive_messages() if m.role=='assistant' and not m.tool_calls]
        print(replies[-1] if replies else '')
        print(f'Session: {store.path}',file=sys.stderr)
    finally:
        try:
            ctx.dispose()
        finally:
            runtime.close()


if __name__ == '__main__':
    main()
