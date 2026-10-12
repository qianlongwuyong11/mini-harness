"""Small, explicitly labelled, single-paper retrieval benchmark. Run each engine in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import statistics
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chunking_eval import atomic_write
from mini_harness.rag.embedding import EmbeddingConfig, LocalE5Embedding

MODELS = {
    "small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", 384),
    "base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", 768),
}
CHUNKS = ROOT / ".mini-harness/rag/chunk-eval/531c10b5362c32281a016d75/chunks.json"


def metrics(ranks, questions):
    result = {}
    for k in (1, 3, 5):
        recalls, hits = [], []
        for row, q in zip(ranks, questions):
            gold = set(q["relevant_ordinals"])
            found = len(gold.intersection(row[:k]))
            recalls.append(found / len(gold))
            hits.append(float(found > 0))
        result[f"recall@{k}"] = statistics.mean(recalls)
        result[f"hit@{k}"] = statistics.mean(hits)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=[*MODELS, "bm25"])
    args = parser.parse_args()
    corpus_bytes = CHUNKS.read_bytes()
    chunks = [c for c in json.loads(corpus_bytes) if c["source_kind"] == "paper"]
    query_bytes = (ROOT / "docs/embedding-queries.json").read_bytes()
    questions = json.loads(query_bytes)
    assert all(set(q["relevant_ordinals"]) <= {c["ordinal"] for c in chunks} for q in questions)
    report = dict(engine=args.engine, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
                  queries_sha256=hashlib.sha256(query_bytes).hexdigest(), corpus_chunks=len(chunks),
                  questions=len(questions), limitations="Single paper, 12 manually curated questions; provisional relevance labels, not a held-out benchmark")
    memory = [psutil.Process().memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            memory.append(psutil.Process().memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    latencies, ranks = [], []
    try:
        if args.engine != "bm25":
            model_id, revision, dimensions = MODELS[args.engine]
            config = EmbeddingConfig(model_id, revision, dimensions)
            started = time.perf_counter()
            service = LocalE5Embedding(config, ROOT / ".mini-harness/rag/models" / model_id.split("/")[-1])
            report["fresh_process_load_seconds_cached_weights"] = time.perf_counter() - started
            print("Model loaded: " + model_id, flush=True)
            import numpy as np
            started = time.perf_counter()
            vectors = np.asarray(service.embed_documents([c["text"] for c in chunks]), dtype=np.float32)
            report["document_encoding_seconds"] = time.perf_counter() - started
            report["document_vectors_per_second"] = len(chunks) / report["document_encoding_seconds"]
            norms = np.linalg.norm(vectors, axis=1)
            assert vectors.shape == (len(chunks), dimensions) and np.allclose(norms, 1, atol=1e-5)
            report["norm_range"] = [float(norms.min()), float(norms.max())]
            search_times = []
            # First query is measured separately, then repeat all questions for a small latency sample.
            started = time.perf_counter()
            first = service.embed_query(questions[0]["query"])
            report["first_query_encoding_ms"] = (time.perf_counter() - started) * 1000
            for repeat in range(3):
                for q in questions:
                    started = time.perf_counter()
                    query = np.asarray(service.embed_query(q["query"]), dtype=np.float32)
                    latencies.append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    indices = np.argsort(-(vectors @ query), kind="stable")[:5]
                    search_times.append((time.perf_counter() - started) * 1000)
                    if repeat == 0:
                        ranks.append([chunks[i]["ordinal"] for i in indices])
            report["exact_vector_search_ms_median"] = statistics.median(search_times)
            report["embedding"] = service.metadata()
            artifact = ROOT / ".mini-harness/rag/embedding-eval" / args.engine
            artifact.mkdir(parents=True, exist_ok=True)
            np.save(artifact / "vectors.npy", vectors)
            atomic_write(artifact / "manifest.json", json.dumps(dict(embedding=service.metadata(),
                corpus_sha256=report["corpus_sha256"], chunk_ids=[c["chunk_id"] for c in chunks]), indent=2))
            report["artifact"] = str(artifact)
        else:
            from mini_harness.rag.es_service import ESService, ESConfig
            from local_es import CONFIG
            service = ESService(ESConfig.load(CONFIG))
            service.health()
            name = "mini-harness-embedding-eval-" + secrets.token_hex(8)
            created = False
            try:
                response = service.client.put("/" + name, json={"settings":{"number_of_shards":1,"number_of_replicas":0},
                    "mappings":{"properties":{"content":{"type":"text","analyzer":"english","similarity":"BM25"}}}})
                response.raise_for_status()
                created = True
                payload = "".join(json.dumps({"index":{"_id":str(c["ordinal"])}}) + "\n" + json.dumps({"content":c["text"]}) + "\n" for c in chunks)
                started = time.perf_counter()
                response = service.client.post("/"+name+"/_bulk",params={"refresh":"wait_for"},content=payload.encode(),headers={"Content-Type":"application/x-ndjson"})
                response.raise_for_status()
                if response.json()["errors"]:
                    raise RuntimeError("BM25 bulk failure")
                report["bulk_and_refresh_seconds"] = time.perf_counter()-started
                for repeat in range(3):
                    for q in questions:
                        started=time.perf_counter()
                        response=service.client.post("/"+name+"/_search",json={"size":5,"query":{"match":{"content":q["query"]}}})
                        response.raise_for_status()
                        latencies.append((time.perf_counter()-started)*1000)
                        if repeat==0:
                            ranks.append([int(h["_id"]) for h in response.json()["hits"]["hits"]])
                response=service.client.get("/_nodes/stats/jvm")
                response.raise_for_status()
                report["es_heap_used_mib_end"] = sum(n["jvm"]["mem"]["heap_used_in_bytes"] for n in response.json()["nodes"].values())/2**20
            finally:
                if created:
                    service.client.delete("/"+name).raise_for_status()
                service.close()
        report["quality"] = metrics(ranks, questions)
        report["pure_chinese_quality"] = metrics(ranks[:8], questions[:8])
        report["mixed_query_quality"] = metrics(ranks[8:], questions[8:])
        report["query_ms_median"] = statistics.median(latencies)
        report["query_ms_p95"] = sorted(latencies)[int(0.95*(len(latencies)-1))]
        report["rankings"] = [dict(q, retrieved_ordinals=row) for q,row in zip(questions,ranks)]
    finally:
        stop.set()
        thread.join()
    report["process_peak_rss_mib"] = max(memory)/2**20
    report["process_baseline_rss_mib"] = memory[0]/2**20
    atomic_write(ROOT / "docs" / ("embedding-eval-"+args.engine+".json"),json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rankings"},ensure_ascii=True),flush=True)


if __name__ == "__main__":
    main()
