"""Actual-model checks for padding invariance and tokenizer compatibility."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embedding_eval import MODELS, CHUNKS
from mini_harness.rag.chunking import EmbeddingTokenizer
from mini_harness.rag.embedding import EmbeddingConfig, LocalE5Embedding

for key in ("small", "base"):
    name, revision, dimensions = MODELS[key]
    path = ROOT / ".mini-harness/rag/models" / name.split("/")[-1]
    service = LocalE5Embedding(EmbeddingConfig(name,revision,dimensions,device="auto"), path)
    texts = ["Queries are initialized as pillars in BEV space.", "Longer paragraph about adaptive sampling. " * 12]
    together = np.asarray(service.embed_documents(texts))
    alone = np.asarray(service.embed_documents([texts[0]]))[0]
    assert np.allclose(together[0],alone,atol=2e-5), "Batch padding changed the embedding"
    counter = EmbeddingTokenizer(path / "tokenizer.json", name)
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    lengths = [len(service.tokenizer("passage: " + c["text"])["input_ids"]) for c in chunks]
    assert lengths == [counter.count(c["text"]) for c in chunks], "Chunk counter differs from model tokenizer"
    try:
        service.embed_query("word " * 600)
        raise AssertionError("Long query was silently truncated")
    except ValueError:
        pass
    print(key, "PASS", "max_tokens",max(lengths), "device",service.device, "fallback",service.fallback_reason,flush=True)
    del service
