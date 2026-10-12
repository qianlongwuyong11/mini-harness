from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mini_harness.rag.embedding import EmbeddingConfig, LocalE5Embedding


class EmbeddingConfigTests(unittest.TestCase):
    def test_rejects_mutable_revision_wrong_dimensions_and_limits(self):
        valid = EmbeddingConfig("intfloat/multilingual-e5-small", "a"*40, 384)
        for change in ({"revision":"main"}, {"dimensions":768}, {"max_tokens":513}, {"batch_size":0}):
            with self.assertRaises(ValueError):
                replace(valid, **change)


class EmbeddingComputationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("rag-embedding optional dependency not installed")
        self.torch = torch
        self.service = LocalE5Embedding.__new__(LocalE5Embedding)
        self.service.config = EmbeddingConfig("intfloat/multilingual-e5-small", "a"*40, 384, max_tokens=16)
        self.service._torch = torch
        self.service.device = "cpu"
        self.service.fallback_reason = None

    def test_pooling_ignores_padding_and_normalizes(self):
        torch = self.torch
        self.service.model = lambda **kwargs: SimpleNamespace(last_hidden_state=torch.tensor([[[3.,4.],[900.,900.]]]))
        result = self.service._forward({"input_ids":torch.tensor([[1,0]]), "attention_mask":torch.tensor([[1,0]])})
        self.assertAlmostEqual(result[0][0], 0.6, places=6)
        self.assertAlmostEqual(result[0][1], 0.8, places=6)

    def test_prefixes_empty_and_overflow(self):
        torch = self.torch
        tokenizer = Mock(return_value={"input_ids":torch.ones((1,3),dtype=torch.long)})
        self.service.tokenizer = tokenizer
        self.service._forward = Mock(return_value=((1.,0.),))
        self.service.embed_query("question")
        self.assertEqual(tokenizer.call_args.args[0], ["query: question"])
        self.service.embed_documents(["document"])
        self.assertEqual(tokenizer.call_args.args[0], ["passage: document"])
        self.assertFalse(tokenizer.call_args.kwargs["truncation"])
        self.assertEqual(self.service.embed_documents([]), ())
        with self.assertRaises(ValueError):
            self.service.embed_query(" ")
        tokenizer.return_value = {"input_ids":torch.ones((1,17),dtype=torch.long)}
        with self.assertRaisesRegex(ValueError, "exceeds token limit"):
            self.service.embed_query("long")

    def test_cuda_oom_retries_on_cpu(self):
        torch = self.torch
        self.service.device = "cuda"
        self.service.model = Mock()
        self.service.tokenizer = Mock(return_value={"input_ids":torch.ones((1,3),dtype=torch.long)})
        self.service._forward = Mock(side_effect=[torch.cuda.OutOfMemoryError(), ((1.,0.),)])
        self.assertEqual(self.service.embed_query("question"), (1.,0.))
        self.assertEqual(self.service.device, "cpu")
        self.service.model.to.assert_called_once_with("cpu")
