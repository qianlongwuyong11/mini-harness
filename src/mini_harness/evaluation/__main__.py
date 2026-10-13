"""CLI for dataset validation and local execution queue (no automatic remote call)."""
import argparse
import json

from .dataset import load_dataset, bind_papers
from .store import EvaluationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=".mini-harness/evals")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "enqueue"):
        sub = commands.add_parser(command)
        sub.add_argument("root")
        sub.add_argument("file")
        sub.add_argument("--catalog", default=".mini-harness/rag/vector-catalog.sqlite3")
    for command in ("status", "cancel", "retry"):
        commands.add_parser(command).add_argument("run_id")
    work = commands.add_parser("work")
    work.add_argument("--catalog", default=".mini-harness/rag/vector-catalog.sqlite3")
    work.add_argument("--allow-model-api", action="store_true",
                      help="Authorize sending queued questions and selected paper evidence to the configured model API")
    args = parser.parse_args()
    store = EvaluationStore(args.store)
    if args.command in ("validate", "enqueue"):
        from ..rag.runtime import PaperRuntime, RagConfig
        dataset = load_dataset(args.root, args.file)
        runtime = PaperRuntime(RagConfig(catalog_path=args.catalog))
        try:
            bind_papers(dataset, runtime.catalog)
        finally:
            runtime.close()
        result = {"dataset_id": dataset.dataset_id, "cases": len(dataset.cases)}
        if args.command == "enqueue":
            from pathlib import Path
            result["run_id"] = store.enqueue(dataset, {"profile": "paper-readonly-v1", "catalog": str(Path(args.catalog).resolve())})
    elif args.command == "status":
        result = {"run": store.get_run(args.run_id), "attempts": [{**attempt, "metrics": store.metrics(attempt["attempt_id"])} for attempt in store.attempts(args.run_id)]}
    elif args.command in ("cancel", "retry"):
        getattr(store, args.command)(args.run_id)
        result = store.get_run(args.run_id)
    else:
        if not args.allow_model_api:
            parser.error("work requires --allow-model-api; questions/evidence will be sent to the model provider")
        from ..client import DeepSeekClient
        from ..rag.runtime import PaperRuntime, RagConfig
        from .runner import EvaluationWorker
        runtime = PaperRuntime(RagConfig(catalog_path=args.catalog))
        try:
            EvaluationWorker(store, DeepSeekClient, runtime).work()
        finally:
            runtime.close()
        result = {"queue": "drained"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
