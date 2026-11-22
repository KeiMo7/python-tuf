# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Transactional local metadata cache.

``MetadataCache`` makes all metadata writes of an :class:`~tuf.ngclient.Updater`
crash-safe and multi-process safe without changing the on-disk metadata layout:

* Writes are staged in a hidden ``.tuf-cache`` state directory and then applied
  with a journaled two-phase commit. A refresh either becomes visible in full
  or not at all: readers never observe a partially written metadata set (each
  individual file is replaced atomically via ``os.replace``).
* Every cache directory has a lock file so that refreshes from multiple
  processes are serialized. Locks are released by the kernel when a process
  dies, so there are no stale locks.
* On startup the cache runs an idempotent recovery: unfinished transactions are
  rolled back to the previous state, committed-but-not-finalized transactions
  are rolled forward, and leftover temporary files are removed. Recovery can be
  interrupted and re-run any number of times with the same result.

The state directory is not part of the TUF metadata layout. Removing it only
discards transaction bookkeeping, never committed metadata: the cache stays
readable with the usual ``*.json`` files, ``root_history`` and ``root.json``
link.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Platform-specific locking primitives. At most one is available; imports are
# guarded so the module loads on every supported operating system.
if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

STATE_DIRNAME = ".tuf-cache"
LOCK_FILENAME = "lock"
TMP_DIRNAME = "tmp"
JOURNAL_FILENAME = "journal.json"
STAGED_DIRNAME = "staged"
BACKUP_DIRNAME = "backup"

_JOURNAL_STATE_PREPARED = "prepared"
_JOURNAL_STATE_COMMITTED = "committed"


class StorageError(OSError):
    """Raised when the cache cannot be kept in a consistent state.

    It is an :class:`OSError` subclass so that callers handling the documented
    "metadata could not be written to disk" errors keep working unchanged.
    """


def _fsync_dir(path: str) -> None:
    """Best-effort fsync of a directory so renames survive a power loss."""
    if not hasattr(os, "O_DIRECTORY"):
        # Directories cannot be opened this way on all platforms (e.g. Windows)
        return  # pragma: no cover - platform without O_DIRECTORY
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - depends on filesystem fsync support
        # fsync is a durability enhancement, not a correctness requirement for
        # the journal protocol: ignore filesystems that do not support it.
        pass
    finally:
        os.close(fd)


class _FileLock:
    """An inter-process lock backed by a lock file.

    POSIX uses ``flock`` (also serializes threads within a process), Windows
    uses ``msvcrt.locking`` with polling. The kernel releases the lock when the
    process exits or dies, so a crashed holder never leaves a stale lock.
    """

    def __init__(self, lock_path: str, timeout: float | None) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._fd: int | None = None

    def acquire(self, shared: bool = False) -> None:
        # "ab" creates the file if needed without truncating it and positions
        # the stream at the end; locking operates on the file description.
        self._fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":  # pragma: no cover - Windows only
                self._acquire_windows(shared)
            else:
                self._acquire_posix(shared)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def _acquire_posix(self, shared: bool) -> None:
        if self._fd is None:  # pragma: no cover - acquire() always opens an fd
            raise StorageError("Cannot lock before opening the lock file")
        fd = self._fd
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if self._timeout is None:
            fcntl.flock(fd, flags)
            return

        # Non-blocking polling so that a configurable timeout is available on
        # every platform while the default remains a blocking lock.
        deadline = time.monotonic() + max(0.0, self._timeout)
        while True:
            try:
                fcntl.flock(fd, flags | fcntl.LOCK_NB)
                return
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise  # pragma: no cover - unexpected flock error
            if time.monotonic() >= deadline:
                raise StorageError(
                    errno.ETIMEDOUT,
                    f"Timed out waiting for cache lock {self._lock_path}",
                )
            time.sleep(0.05)

    def _acquire_windows(self, shared: bool) -> None:  # pragma: no cover
        mode = msvcrt.LK_RLCK if shared else msvcrt.LK_LOCK
        deadline = None
        if self._timeout is not None:
            deadline = time.monotonic() + self._timeout
        while True:
            try:
                # One-byte region lock; LK_LOCK retries internally for ~10s
                msvcrt.locking(self._fd, mode, 1)
                return
            except OSError as e:
                if e.errno != errno.EDEADLK:
                    raise
            if deadline is not None and time.monotonic() >= deadline:
                raise StorageError(
                    errno.ETIMEDOUT,
                    f"Timed out waiting for cache lock {self._lock_path}",
                )
            time.sleep(0.05)

    def release(self) -> None:
        if self._fd is None:
            return  # pragma: no cover - release() is called once after acquire
        try:
            if os.name == "nt":  # pragma: no cover - Windows only
                # Rewind to the locked region before unlocking.
                os.lseek(self._fd, 0, os.SEEK_SET)
                with contextlib.suppress(OSError):
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


class Transaction:
    """A unit of cache writes with atomic commit and crash recovery.

    Create one through :meth:`MetadataCache.transaction` rather than
    instantiating directly. Files are first staged (committed files are never
    touched until :meth:`commit`), journaled, then applied with a sequence of
    atomic renames. Any exception before commit aborts the transaction and
    leaves the cache exactly as it was.
    """

    def __init__(self, cache: MetadataCache) -> None:
        self._cache = cache
        self._dir = cache.metadata_dir
        self._id = uuid.uuid4().hex
        self._txn_dir = os.path.join(cache.state_dir, f"txn-{self._id}")
        self._staged_dir = os.path.join(self._txn_dir, STAGED_DIRNAME)
        self._backup_dir = os.path.join(self._txn_dir, BACKUP_DIRNAME)
        self._journal_path = os.path.join(self._txn_dir, JOURNAL_FILENAME)
        self._entries: list[dict[str, Any]] = []
        self._counter = 0
        self._committed = False
        self._closed = False
        # Becomes True once the prepared journal is durable: from that point
        # on, an interrupted commit must be reconciled via recovery instead of
        # simply deleting the transaction directory.
        self._journal_written = False
        # Directories are created lazily on the first staged write, so a
        # transaction that ends up with no changes touches no disk at all.
        self._txn_dir_created = False
        self._backup_dir_created = False

    def _ensure_txn_dirs(self) -> None:
        if not self._txn_dir_created:
            os.makedirs(self._staged_dir)
            self._txn_dir_created = True

    @property
    def id(self) -> str:
        """Unique transaction identifier (directory name suffix)."""
        return self._id

    @property
    def is_committed(self) -> bool:
        """Whether the transaction has durably committed."""
        return self._committed

    @property
    def is_closed(self) -> bool:
        """Whether the transaction resources have been released."""
        return self._closed

    def _relative_path(self, path: str) -> str:
        """Validate that ``path`` stays inside the metadata directory."""
        root = Path(self._dir).resolve()
        target = Path(self._dir, path).resolve()
        if root != target and root not in target.parents:
            raise StorageError(
                f"Refusing to write outside the metadata directory: {path}"
            )
        return os.path.relpath(target, root)

    def _next_name(self) -> str:
        self._counter += 1
        return str(self._counter)

    def _ensure_backup_dir(self) -> str:
        if not self._backup_dir_created:
            os.makedirs(self._backup_dir, exist_ok=True)
            self._backup_dir_created = True
        return self._backup_dir

    def add_or_replace(self, relative_path: str, data: bytes) -> None:
        """Stage ``data`` to be committed at ``relative_path``.

        If a committed file already exists, its current content is backed up so
        that a crash during commit can be rolled back to it.
        """
        rel = self._relative_path(relative_path)
        name = self._next_name()
        self._ensure_txn_dirs()
        staged = os.path.join(self._staged_dir, name)

        # Write into the shared tmp dir, then atomically move into the
        # transaction's staged directory.
        fd, tmp_path = tempfile.mkstemp(
            dir=self._cache.tmp_dir, prefix=f"stage-{self._id}-"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, staged)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise

        target = os.path.join(self._dir, rel)
        entry: dict[str, Any] = {"op": "add", "path": rel, "staged": name}
        if os.path.lexists(target):
            backup_name = name
            shutil.copy2(
                target,
                os.path.join(self._ensure_backup_dir(), backup_name),
            )
            entry = {
                "op": "replace",
                "path": rel,
                "staged": name,
                "backup": backup_name,
            }
        self._entries.append(entry)

    def update_root_link(self, relative_target: str) -> None:
        """Record that ``root.json`` must point at ``relative_target``.

        The current link target (if any) is recorded so a crash during commit
        can restore it.
        """
        # The link path is fixed by the TUF cache layout and must not be
        # resolved as a relative path.
        link_rel = "root.json"
        target_rel = self._relative_path(relative_target)
        link_path = os.path.join(self._dir, link_rel)
        # Fast path: the link already points at the required target, so there
        # is nothing to do and no journal entry is needed.
        if os.path.islink(link_path) and os.readlink(link_path) == target_rel:
            return
        old_target: str | None = None
        if os.path.islink(link_path):
            old_target = os.readlink(link_path)
        elif os.path.lexists(link_path):
            # Hard-link fallback (Windows without symlink privilege) or a plain
            # file: back the content up like a regular replacement.
            name = self._next_name()
            shutil.copy2(
                link_path, os.path.join(self._ensure_backup_dir(), name)
            )
            old_target = f"backup:{name}"
        self._entries.append(
            {
                "op": "link",
                "path": link_rel,
                "target": target_rel,
                "old_target": old_target,
            }
        )

    def _write_journal(self, state: str, fsync_directory: bool = True) -> None:
        payload = {"state": state, "entries": self._entries}
        fd, tmp_path = tempfile.mkstemp(
            dir=self._txn_dir, prefix="journal-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._journal_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise
        # The prepared journal must be fully durable so a crash during commit
        # can be reconciled. The committed journal only marks that cleanup is
        # pending: if its rename is lost after a crash, recovery sees the
        # prepared journal and rolls back to the still-consistent old state, so
        # an extra directory fsync is unnecessary there.
        if fsync_directory:
            _fsync_dir(self._txn_dir)

    def _apply_link(self, entry: dict[str, Any]) -> None:
        link_path = os.path.join(self._dir, entry["path"])
        target_path = os.path.join(self._dir, entry["target"])
        with contextlib.suppress(FileNotFoundError):
            os.remove(link_path)
        try:
            os.symlink(entry["target"], link_path)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:
                # NTFS without the symlink privilege: use a hard link instead
                os.link(target_path, link_path)
            else:
                raise

    def commit(self) -> None:
        """Apply all staged changes durably and remove the transaction."""
        if self._committed:
            # Idempotent: committing an already committed transaction is a
            # no-op (this must be checked before _closed since a successful
            # commit also closes the transaction).
            return
        if self._closed:
            raise StorageError("Transaction is already closed")

        # An empty transaction (for example an unchanged refresh) has nothing
        # to make durable: mark it committed without creating a journal or
        # touching the committed cache.
        if not self._entries:
            self._committed = True
            self._closed = True
            return

        # Phase 1: persist the "prepared" journal before touching
        # committed files
        self._ensure_txn_dirs()
        self._write_journal(_JOURNAL_STATE_PREPARED)
        self._journal_written = True

        try:
            # Phase 2: apply file operations in journal order, each atomic.
            # Collect the affected parent directories and fsync each only once
            # after all renames have been issued (rather than once per rename),
            # which removes redundant fsyncs while keeping every rename durable.
            touched_dirs: set[str] = set()
            for entry in self._entries:
                target = os.path.join(self._dir, entry["path"])
                if entry["op"] in ("add", "replace"):
                    parent = os.path.dirname(target)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    os.replace(
                        os.path.join(self._staged_dir, entry["staged"]),
                        target,
                    )
                    touched_dirs.add(parent or self._dir)
                else:
                    self._apply_link(entry)
                    touched_dirs.add(self._dir)
            for directory in touched_dirs:
                _fsync_dir(directory)
        except BaseException:
            # Commit was interrupted after the prepared journal was made
            # durable. Keep the transaction directory (journal + backups) so
            # that recover() on the next start rolls it back consistently.
            self._closed = True
            raise

        # Phase 3: mark committed, then clean up. The committed journal does
        # not need a directory fsync: losing this rename just makes recovery
        # treat the transaction as prepared and roll back to the old state.
        self._write_journal(_JOURNAL_STATE_COMMITTED, fsync_directory=False)
        self._committed = True
        self.close()

    def rollback(self) -> None:
        """Discard staged changes without touching committed files.

        Before the prepared journal exists no committed file has been
        modified, so removing the transaction directory is a complete
        rollback. After the journal exists the transaction is reconciled by
        :meth:`MetadataCache.recover` and the directory is intentionally kept.
        """
        if self._closed:
            return
        self._closed = True
        if not self._journal_written:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._txn_dir)

    def close(self) -> None:
        """Release transaction resources (rolls back if not committed)."""
        if not self._committed:
            self.rollback()
            return
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self._txn_dir)


class MetadataCache:
    """Transaction manager for the local metadata directory.

    A single instance owns the hidden state directory and lock for one
    metadata directory. Use :meth:`transaction` for an exclusive, recovered
    write transaction and :meth:`shared_lock` for cooperative readers.
    """

    def __init__(
        self, metadata_dir: str, lock_timeout: float | None = None
    ) -> None:
        self.metadata_dir = metadata_dir
        self._lock_timeout = lock_timeout
        self.state_dir = os.path.join(metadata_dir, STATE_DIRNAME)
        self.tmp_dir = os.path.join(self.state_dir, TMP_DIRNAME)
        self._lock_path = os.path.join(self.state_dir, LOCK_FILENAME)
        self._lock = _FileLock(self._lock_path, lock_timeout)
        self._startup_recovered = False

    def _ensure_state_dirs(self) -> None:
        os.makedirs(self.tmp_dir, exist_ok=True)

    def recover_on_startup(self) -> None:
        """Recover interrupted transactions once when a client starts.

        This runs even when the current session performs no write of its own
        (for example because the cache is already current), so a transaction
        left unfinished by a previous process is always reconciled before the
        cache is used. Safe and idempotent to call repeatedly: only the first
        call takes the lock and runs recovery.
        """
        if self._startup_recovered:
            return
        self._ensure_state_dirs()
        self._lock.acquire(shared=False)
        try:
            self.recover()
            self._startup_recovered = True
        finally:
            self._lock.release()

    @contextlib.contextmanager
    def shared_lock(self) -> Iterator[None]:
        """Take a shared lock: blocks exclusive transactions while held.

        Individual metadata files are always atomically replaced, so readers
        that do not take this lock still never observe torn files; the shared
        lock additionally guarantees that no commit happens while a cooperative
        reader traverses the whole metadata set.
        """
        self._ensure_state_dirs()
        lock = _FileLock(self._lock_path, self._lock_timeout)
        lock.acquire(shared=True)
        try:
            yield
        finally:
            lock.release()

    def recover(self) -> None:
        """Bring the cache to a consistent state after an interrupted run.

        Idempotent: running it repeatedly, including after a crash during
        recovery itself, always yields the same state. Must be called while
        holding the exclusive cache lock.
        """
        if not os.path.isdir(self.state_dir):
            return

        # 1. Remove leftover temporary files (this includes tmp files left by
        # older client versions that used NamedTemporaryFile directly).
        tmp = self.tmp_dir
        if os.path.isdir(tmp):
            for entry in os.scandir(tmp):
                with contextlib.suppress(FileNotFoundError):
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
        else:
            os.makedirs(tmp, exist_ok=True)

        # 2. Resolve every transaction directory found in the state directory
        for name in sorted(os.listdir(self.state_dir)):
            if not name.startswith("txn-"):
                continue
            txn_dir = os.path.join(self.state_dir, name)
            if not os.path.isdir(txn_dir):
                continue
            self._recover_transaction(txn_dir)

    def _recover_transaction(self, txn_dir: str) -> None:
        journal_path = os.path.join(txn_dir, JOURNAL_FILENAME)
        staged_dir = os.path.join(txn_dir, STAGED_DIRNAME)
        backup_dir = os.path.join(txn_dir, BACKUP_DIRNAME)

        try:
            with open(journal_path, encoding="utf-8") as f:
                journal = json.load(f)
            entries = journal["entries"]
            state = journal.get("state")
        except (OSError, KeyError, json.JSONDecodeError) as e:
            # No usable journal: nothing was durably committed, remove the
            # incomplete transaction artifacts.
            logger.debug("Removing transaction without valid journal: %s", e)
            self._safe_rmtree(txn_dir)
            return

        try:
            if state == _JOURNAL_STATE_COMMITTED:
                self._roll_forward(entries, staged_dir)
            else:
                self._roll_back(entries, backup_dir)
        except OSError as e:
            raise StorageError(
                f"Failed to recover metadata cache transaction {txn_dir}: {e}"
            ) from e

        self._safe_rmtree(txn_dir)

    def _roll_forward(
        self, entries: list[dict[str, Any]], staged_dir: str
    ) -> None:
        """Re-apply a committed transaction; every step is idempotent."""
        for entry in entries:
            target = os.path.join(self.metadata_dir, entry["path"])
            if entry["op"] in ("add", "replace"):
                staged = os.path.join(staged_dir, entry["staged"])
                if os.path.exists(staged):
                    parent = os.path.dirname(target)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    os.replace(staged, target)
                # Missing staged means the replacement was already applied
            else:
                target_path = os.path.join(self.metadata_dir, entry["target"])
                current = None
                if os.path.islink(target):
                    current = os.readlink(target)
                if current != entry["target"] and os.path.exists(target_path):
                    self._apply_recovered_link(target, entry["target"])

    def _roll_back(
        self, entries: list[dict[str, Any]], backup_dir: str
    ) -> None:
        """Undo a transaction that never reached the committed state."""
        # Reverse order so the last applied operation is undone first
        for entry in reversed(entries):
            target = os.path.join(self.metadata_dir, entry["path"])
            if entry["op"] == "add":
                # Only remove if the file is not a pre-existing directory entry
                # we did not create: adds had no backup, so lexists == ours.
                if os.path.lexists(target) and not os.path.isdir(target):
                    os.remove(target)
            elif entry["op"] == "replace":
                backup = os.path.join(backup_dir, entry["backup"])
                if os.path.exists(backup):
                    os.replace(backup, target)
                # Missing backup means the rename had not happened yet: the
                # original committed file is untouched.
            else:
                old_target = entry.get("old_target")
                if old_target is None:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(target)
                elif old_target.startswith("backup:"):
                    backup = os.path.join(
                        backup_dir, old_target.split(":", 1)[1]
                    )
                    if os.path.exists(backup):
                        os.replace(backup, target)
                else:
                    self._apply_recovered_link(target, old_target)

    def _apply_recovered_link(
        self, link_path: str, relative_target: str
    ) -> None:
        target_path = os.path.join(self.metadata_dir, relative_target)
        with contextlib.suppress(FileNotFoundError):
            os.remove(link_path)
        try:
            os.symlink(relative_target, link_path)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:  # pragma: no cover
                os.link(target_path, link_path)
            else:
                raise

    @staticmethod
    def _safe_rmtree(path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(path)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """Open an exclusive, crash-recovered write transaction.

        Takes the cache lock, runs recovery for any interrupted predecessor,
        then yields a fresh :class:`Transaction`. On any exception the
        transaction is rolled back and the cache keeps its previous state.
        """
        self._ensure_state_dirs()
        self._lock.acquire(shared=False)
        txn: Transaction | None = None
        try:
            if not self._startup_recovered:
                self.recover()
            txn = Transaction(self)
            yield txn
        except BaseException:
            if txn is not None:
                txn.rollback()
            raise
        else:
            if not txn.is_committed and not txn.is_closed:
                # A context body that returns without committing is an abort
                txn.rollback()
        finally:
            self._lock.release()
