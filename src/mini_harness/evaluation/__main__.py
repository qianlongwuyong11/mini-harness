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
        if command == "enqueue":
            sub.add_argument("--judge-config", help="JSON config enabling live_judge mode")
    for command in ("status", "cancel", "retry"):
        commands.add_parser(command).add_argument("run_id")
    work = commands.add_parser("work")
    work.add_argument("--catalog", default=".mini-harness/rag/vector-catalog.sqlite3")
    work.add_argument("--allow-model-api", action="store_true",
                      help="Authorize sending queued questions and selected paper evidence to the configured model API")
    work.add_argument("--allow-judge-api", action="store_true")
    rescore = commands.add_parser("rescore")
    rescore.add_argument("run_id")
    rescore.add_argument("--judge-config", required=True)
    rescore.add_argument("--allow-judge-api", action="store_true")
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
            configuration = {"profile": "paper-readonly-v1", "catalog": str(Path(args.catalog).resolve())}
            if args.judge_config:
                configuration["judge"] = json.loads(Path(args.judge_config).read_text(encoding="utf-8-sig"))
            result["run_id"] = store.enqueue(dataset, configuration, mode="live_judge" if args.judge_config else "live_rules")
    elif args.command == "status":
        result = {"run": store.get_run(args.run_id), "attempts": [{**attempt, "metrics": store.metrics(attempt["attempt_id"]), "scores": store.scores(attempt["attempt_id"])} for attempt in store.attempts(args.run_id)]}
    elif args.command == "rescore":
        if not args.allow_judge_api:
            parser.error("rescore requires --allow-judge-api; questions, answers, references and evidence are sent to the configured judge")
        from pathlib import Path
        from .scoring import SemanticScorer
        store.get_run(args.run_id)
        scorer = SemanticScorer(store, json.loads(Path(args.judge_config).read_text(encoding="utf-8-sig")), allow_api=True)
        with store.worker_lock():
            store.recover()
            latest = {a["case_id"]: a for a in store.attempts(args.run_id)}
            result = {"scores": [scorer.score(a) for a in latest.values() if a["status"] == "completed"]}
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
            EvaluationWorker(store, DeepSeekClient, runtime, allow_judge_api=args.allow_judge_api).work()
        finally:
            runtime.close()
        result = {"queue": "drained"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
