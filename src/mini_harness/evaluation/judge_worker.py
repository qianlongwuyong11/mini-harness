"""JSON-only isolated scoring process; no login/upload/evaluate orchestration."""
import contextlib
import json
import os
from pathlib import Path
import sys
from .backend import load_local_backend
from .trace import TraceStore


def main():
    payload = json.loads(sys.stdin.read())
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            load_local_backend()
            from .judge import create_judge
            from .deepeval_adapter import score_payload
            config = payload["config"]
            traces = TraceStore(Path(payload["artifact_root"]), secrets=(os.environ.get(config["api_key_env"]),))
            judge = create_judge(config, traces)
            report = score_payload(payload["input"], config, judge)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        from .trace import redact
        print(json.dumps(redact({"error_type": type(error).__name__, "message": str(error)}, TraceStore().secrets)))
        sys.exit(1)


if __name__ == "__main__":
    main()
