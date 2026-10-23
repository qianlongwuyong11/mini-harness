import unittest

from mini_harness.memory.contracts import (
    CommitResult, MemoryBackend, MemoryBackendError, MemoryDocument, MemoryErrorCode,
    MemoryMessage, MemoryScope, TaskState,
)
from memory_fake import FakeMemoryBackend


class MemoryContractTests(unittest.TestCase):
    def test_commit_acceptance_does_not_publish_memory(self):
        fake = FakeMemoryBackend()
        backend: MemoryBackend = fake
        scope = MemoryScope("alice", "aurora")
        backend.append(scope, "session-1", (MemoryMessage("msg-1", "user", "Use SQLite"),))
        receipt = backend.commit(scope, "session-1")
        self.assertEqual(receipt.status, "accepted")
        self.assertEqual(backend.get_task(scope, receipt.task_id).state, TaskState.PENDING)
        self.assertEqual(backend.recall(scope, "SQLite"), ())
        fake.complete(scope, receipt.task_id, (MemoryDocument("fake://decision", "Use SQLite"),))
        self.assertEqual(backend.get_task(scope, receipt.task_id).state, TaskState.COMPLETED)
        hit, = backend.recall(scope, "SQLite")
        self.assertEqual(backend.read(scope, hit.uri).content, "Use SQLite")
        self.assertEqual(backend.commit(scope, "session-1"), CommitResult("session-1", "noop"))

    def test_scope_applies_to_read_recall_session_and_task(self):
        fake = FakeMemoryBackend()
        owner = MemoryScope("alice", "aurora")
        fake.seed(owner, MemoryDocument("fake://decision", "Use SQLite"))
        fake.append(owner, "session-1", (MemoryMessage("msg-1", "user", "Use SQLite"),))
        task_id = fake.commit(owner, "session-1").task_id
        for outsider in (MemoryScope("alice", "other"), MemoryScope("bob", "aurora")):
            with self.subTest(scope=outsider):
                self.assertEqual(fake.recall(outsider, "SQLite"), ())
                for operation in (lambda: fake.read(outsider, "fake://decision"),
                                  lambda: fake.get_task(outsider, task_id),
                                  lambda: fake.commit(outsider, "session-1")):
                    with self.assertRaises(MemoryBackendError) as caught:
                        operation()
                    self.assertEqual(caught.exception.code, MemoryErrorCode.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
