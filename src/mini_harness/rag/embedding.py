"""Local, revision-pinned E5 embeddings. No model imports until construction."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    revision: str
    dimensions: int
    max_tokens: int = 512
    batch_size: int = 4
    device: str = "cpu"
    num_threads: int = 4

    def __post_init__(self):
        expected = {"intfloat/multilingual-e5-small": 384, "intfloat/multilingual-e5-base": 768}
        if expected.get(self.model_id) != self.dimensions:
            raise ValueError("Unsupported E5 model or dimensions")
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Use an immutable 40-character model revision")
        if not 1 <= self.max_tokens <= 512 or self.batch_size < 1 or self.num_threads < 1:
            raise ValueError("Invalid embedding limits")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")


class LocalE5Embedding:
    def __init__(self, config: EmbeddingConfig, model_path: str | Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.config = config
        self._torch = torch
        self.path = Path(model_path)
        # hf_hub_download writes the resolved commit into each local file's metadata.
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            metadata = self.path / ".cache/huggingface/download" / (name + ".metadata")
            if not metadata.exists() or metadata.read_text().splitlines()[0] != config.revision:
                raise ValueError(f"Local artifact revision mismatch: {name}")
        torch.set_num_threads(config.num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True).eval()
        if self.model.config.hidden_size != config.dimensions:
            raise ValueError("Loaded model dimensions differ from configuration")
        self.device = "cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu"
        self.fallback_reason = "CUDA unavailable" if config.device != "cpu" and self.device == "cpu" else None
        try:
            self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            self._cpu_fallback()

    @property
    def model_id(self):
        return self.config.model_id

    @property
    def dimensions(self):
        return self.config.dimensions

    def _cpu_fallback(self):
        self.model.to("cpu")
        self.device = "cpu"
        self.fallback_reason = "CUDA out of memory"
        self._torch.cuda.empty_cache()

    def metadata(self):
        return dict(asdict(self.config), effective_device=self.device, fallback_reason=self.fallback_reason,
                    tokenizer_sha256=hashlib.sha256((self.path / "tokenizer.json").read_bytes()).hexdigest(),
                    pooling="attention-mask mean", normalize="L2", dtype="float32",
                    document_prefix="passage: ", query_prefix="query: ")

    def _forward(self, encoded):
        torch = self._torch
        with torch.inference_mode():
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            hidden = self.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) < 0.99).any():
                raise ValueError("Invalid embedding output")
            return tuple(tuple(row) for row in vectors.cpu().tolist())

    def _embed(self, texts, prefix):
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding input must be nonempty text")
        output = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [prefix + text for text in texts[start:start+self.config.batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > self.config.max_tokens:
                raise ValueError("Embedding input exceeds token limit; rechunk instead of truncating")
            try:
                rows = self._forward(encoded)
            except self._torch.cuda.OutOfMemoryError:
                if self.device == "cpu":
                    raise
                self._cpu_fallback()
                rows = self._forward(encoded)
            output.extend(rows)
        return tuple(output)

    def embed_documents(self, texts):
        return self._embed(texts, "passage: ")

    def embed_query(self, text):
        return self._embed([text], "query: ")[0]


def load_default_config() -> EmbeddingConfig:
    return EmbeddingConfig(**json.loads(Path(__file__).with_name("embedding-default.json").read_text(encoding="utf-8")))
