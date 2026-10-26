"""Opt-in Session + successful-save attachment for automatic memory writes."""
from .background import MemoryBackgroundRunner
from .generation import scan_saved_session


class AutomaticMemorySync:
    """Attach to the SAME JsonlStore used by the session persistence owner.

    Session callbacks only wake the runner. Background reads saved files, never
    live Session events. Known file is rescanned at startup and on each tick, so
    save/notification loss is recoverable. One writer/store per path is required.
    """
    def __init__(self, session, session_store, db_path, backend_factory, *,
                 target_id, scope, **runner_options):
        self.last_scan = None
        self.scan_error = None
        def scan(jobs):
            try:
                with session_store.save_lock:
                    if not session_store.path.exists(): return
                    self.last_scan = scan_saved_session(session_store.path,jobs,
                        target_id=target_id,scope=scope)
                self.scan_error = None
            except Exception:
                # Already durable jobs can still progress; damaged history is not skipped.
                self.scan_error = "saved_session_scan_failed"
        self.runner = MemoryBackgroundRunner(db_path,backend_factory,target_id=target_id,
            scope=scope,before_scan=scan,**runner_options)
        self._unsubscribe_event = session.subscribe(
            lambda event: self.runner.wake() if event.type == "turn/end" else None)
        self._unsubscribe_saved = session_store.subscribe_saved(self.runner.wake)
        try:
            self.runner.start()
        except BaseException:
            self._unsubscribe_event(); self._unsubscribe_saved()
            raise

    def close(self, timeout=5):
        self._unsubscribe_event(); self._unsubscribe_saved()
        return self.runner.stop(timeout)


from dataclasses import dataclass
from ..cordis import depends
from .contracts import MemoryScope


@dataclass(frozen=True)
class AutomaticMemoryConfig:
    session_store: object
    db_path: object
    backend_factory: object
    target_id: str
    scope: MemoryScope


@depends("session")
def automatic_memory_plugin(ctx, config):
    if not isinstance(config, AutomaticMemoryConfig):
        raise ValueError("AutomaticMemoryConfig required")
    sync = AutomaticMemorySync(ctx.session, config.session_store, config.db_path,
        config.backend_factory,target_id=config.target_id,scope=config.scope)
    ctx.provide("memory_sync",sync)
    ctx.effect(lambda: sync.close)
