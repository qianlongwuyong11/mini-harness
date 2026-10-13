"""Local DeepEval bootstrap, exclusively for the isolated scoring process.

Do not import DeepEval in Agent/UI. Stage 5 will launch this worker boundary.
"""
import importlib
import importlib.metadata
import os
import sys

DEEPEVAL_VERSION = "4.2.8"
LOCAL_FLAGS = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_TELEMETRY_ENABLED": "0",
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_DISABLE_LEGACY_KEYFILE": "1",
    "DEEPEVAL_UPDATE_WARNING_OPT_IN": "0",
    "DEEPEVAL_FILE_SYSTEM": "READ_ONLY",
    "DEEPEVAL_MODE": "stable",
    "DEEPEVAL_EVAL_MODE": "llm",
    "DEEPEVAL_NO_INSPECT_PROMPT": "1",
    "CONFIDENT_OPEN_BROWSER": "0",
}

def worker_environment(source=None):
    """Return a copy, never mutate the parent Agent's environment."""
    env = dict(os.environ if source is None else source)
    for key in list(env):
        if key.upper().startswith(("CONFIDENT_", "DEEPEVAL_")):
            del env[key]
    env.update(LOCAL_FLAGS)
    return env

def load_local_backend():
    """Fail closed if SDK was imported before local settings were installed.

Uses direct metric.measure in future adapters; do not login/instrument/upload.
READ_ONLY disables SDK persistence: mini-harness owns local reports.
"""
    if "deepeval" in sys.modules:
        raise RuntimeError("DeepEval already imported; use a fresh scoring process")
    if any(k.upper().startswith("CONFIDENT_") and k.upper() != "CONFIDENT_OPEN_BROWSER" and v
           for k, v in os.environ.items()):
        raise RuntimeError("Use worker_environment before launching the scoring process")
    if importlib.metadata.version("deepeval") != DEEPEVAL_VERSION:
        raise RuntimeError("Unexpected DeepEval version; install the scoring lock file")
    os.environ.update(LOCAL_FLAGS)
    return importlib.import_module("deepeval")

def require_judge(judge):
    """Never let a metric silently instantiate its default remote judge."""
    if judge is None or isinstance(judge, str):
        raise ValueError("An explicitly configured judge adapter is required")
    from deepeval.models import DeepEvalBaseLLM
    if not isinstance(judge, DeepEvalBaseLLM):
        raise TypeError("judge must implement DeepEvalBaseLLM")
    return judge
