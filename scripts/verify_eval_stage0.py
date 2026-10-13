"""Offline SDK verification. Run using .venv-eval; no Agent/judge requests."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from mini_harness.evaluation.backend import worker_environment, load_local_backend, require_judge

def main():
    env = worker_environment()
    for key in list(os.environ):
        if key.upper().startswith(("DEEPEVAL_", "CONFIDENT_")):
            del os.environ[key]
    os.environ.update(env)
    connections = []
    def blocked(*args, **kwargs):
        connections.append("attempt")
        raise RuntimeError("Network prohibited in stage-0 verification")
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        # Conflicting dotenv and legacy keys must not be read.
        Path(".env").write_text("CONFIDENT_API_KEY=fixture-never-real\n",encoding="utf-8")
        Path(".deepeval").mkdir()
        Path(".deepeval/.deepeval").write_text('{"confident_api_key":"fixture-never-real"}',encoding="utf-8")
        before = {str(p):p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
        try:
            with patch.object(socket.socket,"connect",blocked), patch.object(socket.socket,"connect_ex",blocked), patch("socket.create_connection",blocked):
                sdk = load_local_backend()
                from deepeval.config.settings import get_settings
                from deepeval.models import DeepEvalBaseLLM
                from deepeval.metrics import ToolCorrectnessMetric
                from deepeval.test_case import LLMTestCase, ToolCall
                class NoNetworkJudge(DeepEvalBaseLLM):
                    def load_model(self): return None
                    def get_model_name(self): return "offline-fixture"
                    def generate(self,*args,**kwargs): raise AssertionError("Unexpected judge request")
                    async def a_generate(self,*args,**kwargs): raise AssertionError("Unexpected judge request")
                metric=ToolCorrectnessMetric(model=require_judge(NoNetworkJudge()),async_mode=False,include_reason=False)
                call=ToolCall(name="list_papers",input_parameters={})
                good=metric.measure(LLMTestCase(input="List papers",actual_output="listed",tools_called=[call],expected_tools=[call]),_show_indicator=False)
                bad=metric.measure(LLMTestCase(input="List papers",actual_output="none",tools_called=[],expected_tools=[call]),_show_indicator=False)
                settings=get_settings()
                assert not settings.CONFIDENT_API_KEY
                assert settings.DEEPEVAL_TELEMETRY_OPT_OUT
                assert good == 1 and bad == 0
                assert not connections, "SDK attempted a network connection"
                after={str(p):p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
                assert before == after, "SDK wrote local files despite read-only mode"
                print(json.dumps({"deepeval":sdk.__version__,"good_score":good,"bad_score":bad,
                    "network_attempts":len(connections),"sdk_files_changed":False,"cloud_key_loaded":False}))
        finally:
            os.chdir(original)

if __name__ == "__main__": main()
