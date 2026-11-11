"""Shared, reentrant execution locks and transactional execution decorators."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator


class ExecutionBusyError(RuntimeError):
    """Another process owns this portfolio's execution lock."""


_registry_guard = threading.Lock()
_registry: dict[str, tuple[Any, Any]] = {}


@contextmanager
def portfolio_execution_lock(db_path: Path, timeout: float = 30.0) -> Iterator[None]:
    """Serialize all writers, including different storage instances and processes.

    The OS releases the byte lock on process exit. It is independent of SQLite
    transactions so an order intent can commit before an external request.
    """
    lock_path = Path(str(db_path.resolve()) + ".execution.lock")
    key = os.path.normcase(str(lock_path))
    with _registry_guard:
        thread_lock, state = _registry.setdefault(key, (threading.RLock(), threading.local()))
    if not thread_lock.acquire(timeout=timeout):
        raise ExecutionBusyError("Başka bir işlem portföyü güncelliyor; bu adım bekletildi.")
    handle = None
    locked = False
    try:
        depth = getattr(state, "depth", 0)
        if depth:
            state.depth = depth + 1
            try:
                yield
            finally:
                state.depth -= 1
            return
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise ExecutionBusyError("Başka bir uygulama emir yürütüyor; bu adım bekletildi.")
                time.sleep(0.05)
        state.depth = 1
        try:
            yield
        finally:
            state.depth = 0
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        thread_lock.release()


def serialized_execution(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.storage.execution_lock():
            return method(self, *args, **kwargs)
    return wrapped


def atomic_execution(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.storage.execution_lock(), self.storage.atomic():
            return method(self, *args, **kwargs)
    return wrapped
