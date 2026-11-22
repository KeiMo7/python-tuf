# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for transactional, crash-safe and concurrent metadata caching.

These tests exercise the failure-injection, recovery, cancellation and
inter-process locking behavior of
``tuf.ngclient._internal.metadata_cache`` as used by ``Updater``. They
complement the existing update-workflow tests: the point here is that a
refresh is all-or-nothing, an interrupted refresh is reconciled on the next
startup, and concurrent refreshes of the same cache cannot overwrite each
other.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from tests.repository_simulator import RepositorySimulator
from tuf.api.exceptions import (
    BadVersionNumberError,
    ExpiredMetadataError,
    LengthOrHashMismatchError,
    UnsignedMetadataError,
)
from tuf.api.metadata import (
    SPECIFICATION_VERSION,
    DelegatedRole,
    Targets,
    Timestamp,
)
from tuf.ngclient import Updater, UpdaterConfig
from tuf.ngclient._internal.metadata_cache import (
    BACKUP_DIRNAME,
    JOURNAL_FILENAME,
    STAGED_DIRNAME,
    STATE_DIRNAME,
    MetadataCache,
    StorageError,
    Transaction,
)

SPEC_VER = ".".join(SPECIFICATION_VERSION)


class TransactionalRefreshTest(unittest.TestCase):
    """End-to-end refresh failure injection through the public Updater API."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        self.sim = RepositorySimulator()

    def tearDown(self) -> None:
        # Make sure a read-only directory can be cleaned up on platforms
        # where the tests may have made it non-writable.
        for root, dirs, _files in os.walk(self.temp_dir.name):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o700)
        os.chmod(self.metadata_dir, 0o700) if os.path.isdir(
            self.metadata_dir
        ) else None
        self.temp_dir.cleanup()

    def _updater(self, config: UpdaterConfig | None = None) -> Updater:
        return Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            self.sim,
            config=config,
            bootstrap=self.sim.signed_roots[0],
        )

    def _baseline_snapshot(self) -> dict[str, bytes]:
        """Content of every committed metadata file after a good refresh."""
        snapshot: dict[str, bytes] = {}
        for root, _dirs, files in os.walk(self.metadata_dir):
            if STATE_DIRNAME in root.split(os.sep):
                continue
            for name in files:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    snapshot[os.path.relpath(path, self.metadata_dir)] = (
                        os.readlink(path).encode()
                    )
                else:
                    with open(path, "rb") as f:
                        snapshot[os.path.relpath(path, self.metadata_dir)] = (
                            f.read()
                        )
        return snapshot

    def _assert_no_transaction_left(self) -> None:
        state_dir = os.path.join(self.metadata_dir, STATE_DIRNAME)
        # The lock file and an empty tmp directory may remain, but no txn dirs
        if os.path.isdir(state_dir):
            for name in os.listdir(state_dir):
                self.assertFalse(
                    name.startswith("txn-"),
                    f"transaction directory {name} was not cleaned up",
                )
            tmp = os.path.join(state_dir, "tmp")
            if os.path.isdir(tmp):
                self.assertEqual(os.listdir(tmp), [])

    def _assert_cache_equals(self, snapshot: dict[str, bytes]) -> None:
        self.assertEqual(self._baseline_snapshot(), snapshot)

    # ------------------------------------------------------------------
    # Form 1: root role update succeeds, timestamp role fails
    # ------------------------------------------------------------------
    def test_root_ok_timestamp_failure_rolls_back(self) -> None:
        self.sim.root.version += 1
        self.sim.publish_root()
        baseline = self._baseline_snapshot()
        self.assertEqual(baseline, {})  # nothing committed before refresh

        self.sim.signers[Timestamp.type].clear()
        with self.assertRaises(UnsignedMetadataError):
            self._updater().refresh()

        # Only the bootstrap root transaction (from Updater.__init__) is
        # visible: the failed refresh committed nothing new.
        files = self._baseline_snapshot()
        self.assertIn("root.json", files)
        self.assertNotIn("timestamp.json", files)
        self.assertNotIn("snapshot.json", files)
        self.assertNotIn("targets.json", files)
        # The rotated root v2 was staged but must not have been committed
        self.assertFalse(
            os.path.exists(
                os.path.join(self.metadata_dir, "root_history", "2.root.json")
            )
        )
        self._assert_no_transaction_left()

    # ------------------------------------------------------------------
    # Form 2: snapshot succeeds, targets fails
    # ------------------------------------------------------------------
    def test_snapshot_ok_targets_failure_rolls_back(self) -> None:
        self._updater().refresh()
        baseline = self._baseline_snapshot()

        # Publish a genuinely new targets/snapshot/timestamp set which is
        # unsigned so verification fails at the targets stage after timestamp
        # and snapshot have been staged.
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.sim.signers[Targets.type].clear()

        with self.assertRaises(UnsignedMetadataError):
            self._updater().refresh()

        # The whole second refresh is undone: cache is byte-identical
        self._assert_cache_equals(baseline)
        self._assert_no_transaction_left()

    # ------------------------------------------------------------------
    # Form 3 / failure injection at every commit stage: a process killed while
    # applying the transaction is modeled by raising out of the os.replace
    # calls that move staged files into their committed location.
    # ------------------------------------------------------------------
    @staticmethod
    def _commit_destinations(md_dir: str, dst: str) -> bool:
        """True if dst is a committed metadata file (not the state directory)."""
        state_abs = os.path.abspath(os.path.join(md_dir, STATE_DIRNAME))
        dst_abs = os.path.abspath(dst)
        return dst_abs.startswith(
            os.path.abspath(md_dir)
        ) and not dst_abs.startswith(state_abs)

    @staticmethod
    def _snapshot_of(cache_dir: str) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for root, _dirs, files in os.walk(cache_dir):
            if STATE_DIRNAME in root.split(os.sep):
                continue
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, cache_dir)
                if os.path.islink(path):
                    result[rel] = os.readlink(path).encode()
                else:
                    with open(path, "rb") as f:
                        result[rel] = f.read()
        return result

    def _new_updater(self, md_dir: str, tg_dir: str) -> Updater:
        return Updater(
            md_dir,
            "https://example.com/metadata/",
            tg_dir,
            "https://example.com/targets/",
            self.sim,
            bootstrap=self.sim.signed_roots[0],
        )

    def test_failure_injected_at_each_commit_stage(self) -> None:
        # Publish a refresh that changes timestamp, snapshot and targets
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        real_replace = os.replace

        # In a throwaway cache, count the committed-destination replacements
        # one refresh performs (constructor writes happen outside the patch).
        probe_dir = tempfile.mkdtemp(dir=self.temp_dir.name)
        probe_md = os.path.join(probe_dir, "metadata")
        probe_tg = os.path.join(probe_dir, "targets")
        commit_targets: list[str] = []
        probe = self._new_updater(probe_md, probe_tg)

        def counting_replace(src: str, dst: str, *args: object) -> None:
            if self._commit_destinations(probe_md, dst):
                commit_targets.append(dst)
            real_replace(src, dst, *args)

        with patch("os.replace", side_effect=counting_replace):
            probe.refresh()
        self.assertTrue(commit_targets)

        for stage in range(1, len(commit_targets) + 1):
            with self.subTest(stage=stage):
                self._assert_commit_failure_recovers(stage, real_replace)

    def _assert_commit_failure_recovers(
        self, stage: int, real_replace: object
    ) -> None:
        cache_dir = tempfile.mkdtemp(dir=self.temp_dir.name)
        md_dir = os.path.join(cache_dir, "metadata")
        tg_dir = os.path.join(cache_dir, "targets")

        updater = self._new_updater(md_dir, tg_dir)
        before = self._snapshot_of(md_dir)
        counter = {"n": 0}

        def failing_replace(
            src: str,
            dst: str,
            *args: object,
            md_dir: str = md_dir,
            stage: int = stage,
            counter: dict[str, int] = counter,
        ) -> None:
            if self._commit_destinations(md_dir, dst):
                counter["n"] += 1
                if counter["n"] == stage:
                    raise OSError("injected commit failure")
            real_replace(src, dst, *args)

        with (
            patch("os.replace", side_effect=failing_replace),
            self.assertRaises(OSError),
        ):
            updater.refresh()

        # A fresh updater runs recovery in its constructor and brings the
        # cache back to its pre-refresh, consistent state.
        self._new_updater(md_dir, tg_dir)
        self.assertEqual(self._snapshot_of(md_dir), before)
        state = os.path.join(md_dir, STATE_DIRNAME)
        self.assertFalse(any(n.startswith("txn-") for n in os.listdir(state)))

    # ------------------------------------------------------------------
    # Form 4: disk full / no write permission while staging
    # ------------------------------------------------------------------
    def test_disk_full_during_staging_keeps_old_state(self) -> None:
        self._updater().refresh()
        baseline = self._baseline_snapshot()

        # Publish changes so the next refresh genuinely has metadata to write;
        # an unchanged refresh performs no writes at all.
        self.sim.targets.version += 1
        self.sim.update_snapshot()

        # Simulate ENOSPC while staging the first metadata file: this is where
        # a full disk or an unwritable directory surfaces.
        def fail_with_enospc(*_args: object) -> None:
            raise OSError(28, "No space left on device")

        with (
            patch.object(Transaction, "add_or_replace", fail_with_enospc),
            self.assertRaises(OSError),
        ):
            self._updater().refresh()

        self._assert_cache_equals(baseline)
        self._assert_no_transaction_left()

    def test_readonly_metadata_dir_raises_oserror(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory permission bits")
        self._updater().refresh()
        baseline = self._baseline_snapshot()

        # Publish changes so the next refresh genuinely has to write; an
        # unchanged refresh performs no writes and would not hit the wall.
        self.sim.targets.version += 1
        self.sim.update_snapshot()

        os.chmod(self.metadata_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            with self.assertRaises(OSError):
                self._updater().refresh()
        finally:
            os.chmod(self.metadata_dir, 0o700)

        self._assert_cache_equals(baseline)

    # ------------------------------------------------------------------
    # Form 5: two processes (threads here) refreshing the same cache
    # ------------------------------------------------------------------
    def test_concurrent_refreshes_do_not_overwrite(self) -> None:
        self.sim.targets.version += 1
        self.sim.update_snapshot()

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        original = MetadataCache.transaction

        def synchronized(
            self: MetadataCache,
        ) -> object:  # type: ignore[override]
            barrier.wait()
            return original(self)

        def worker() -> None:
            try:
                self._updater().refresh()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        with patch.object(MetadataCache, "transaction", synchronized):
            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        self.assertEqual(errors, [])
        # Exactly one coherent final state, readable by a fresh updater
        updater = self._updater()
        updater.refresh()
        self.assertEqual(
            updater._trusted_set.targets.version, self.sim.targets.version
        )
        self._assert_no_transaction_left()

    def test_lock_timeout_raises_storage_error(self) -> None:
        cache = MetadataCache(self.metadata_dir, lock_timeout=0.1)
        with cache.shared_lock():
            rival = MetadataCache(self.metadata_dir, lock_timeout=0.1)
            with (
                self.assertRaises(StorageError),
                rival.transaction(),
            ):
                self.fail("lock should not have been acquired")

    # ------------------------------------------------------------------
    # Form 6: refresh canceled mid-flight leaves nothing behind
    # ------------------------------------------------------------------
    def test_cancelled_refresh_leaves_no_temp_files(self) -> None:
        self._updater().refresh()
        baseline = self._baseline_snapshot()

        def cancel(*_args: object) -> None:
            raise KeyboardInterrupt

        with (
            patch.object(Updater, "_load_snapshot", cancel),
            self.assertRaises(KeyboardInterrupt),
        ):
            self._updater().refresh()

        self._assert_cache_equals(baseline)
        self._assert_no_transaction_left()

    # ------------------------------------------------------------------
    # Form 7: old root versions are retained for rollback/freeze detection
    # ------------------------------------------------------------------
    def test_root_history_is_retained_for_rollback(self) -> None:
        self.sim.root.version += 1
        self.sim.publish_root()
        self.sim.root.version += 1
        self.sim.publish_root()
        self._updater().refresh()

        history = os.path.join(self.metadata_dir, "root_history")
        self.assertEqual(
            sorted(os.listdir(history)),
            ["1.root.json", "2.root.json", "3.root.json"],
        )

    def test_rollback_detection_still_works_after_refresh(self) -> None:
        self.sim.timestamp.version = 2
        self._updater().refresh()

        # Repository tries to roll the timestamp back
        self.sim.timestamp.version = 1
        with self.assertRaises(BadVersionNumberError):
            self._updater().refresh()

    def test_freeze_detection_still_works_after_refresh(self) -> None:
        self._updater().refresh()

        # Publish an expired timestamp: the freeze-attack check in the
        # verification layer must still reject it even though caching is
        # transactional.
        self.sim.timestamp.expires = datetime.now(UTC) - timedelta(days=1)
        self.sim.update_timestamp()
        with self.assertRaises(ExpiredMetadataError):
            self._updater().refresh()

    # ------------------------------------------------------------------
    # Form 8: already cached delegated role metadata survives a refresh
    # ------------------------------------------------------------------
    def test_cached_delegated_role_is_preserved(self) -> None:
        targets = Targets(1, SPEC_VER, self.sim.safe_expiry, {}, None)
        role = DelegatedRole("role1", [], 1, False, ["*"], None)
        self.sim.add_delegation("targets", role, targets)
        self.sim.update_snapshot()

        updater = self._updater()
        updater.get_targetinfo("non_existent_target")
        delegated_path = os.path.join(self.metadata_dir, "role1.json")
        self.assertTrue(os.path.exists(delegated_path))
        before = self._baseline_snapshot()

        # Refresh again with no repository change: delegated file must remain
        updater2 = self._updater()
        updater2.get_targetinfo("non_existent_target")
        self.assertEqual(self._baseline_snapshot(), before)

    # ------------------------------------------------------------------
    # Form 9: root.json link always points at the current root
    # ------------------------------------------------------------------
    def test_root_link_points_at_current_version(self) -> None:
        self.sim.root.version += 1
        self.sim.publish_root()
        self._updater().refresh()
        link = os.path.join(self.metadata_dir, "root.json")
        self.assertEqual(
            os.readlink(link), os.path.join("root_history", "2.root.json")
        )

    # ------------------------------------------------------------------
    # Form 11: repository metadata changes during refresh is rejected
    # ------------------------------------------------------------------
    def test_version_change_mid_refresh_is_rejected(self) -> None:
        # Pin snapshot hashes in timestamp, then establish a good baseline.
        self.sim.compute_metafile_hashes_length = True
        self.sim.update_timestamp()
        self._updater().refresh()
        baseline = self._baseline_snapshot()

        # Change snapshot contents without updating the timestamp-pinned hash:
        # this is the mix-and-match / version-change the client must reject.
        self.sim.snapshot.expires = self.sim.snapshot.expires + timedelta(
            days=1
        )
        self.sim.snapshot.version += 1
        self.sim.timestamp.snapshot_meta.version = self.sim.snapshot.version
        self.sim.timestamp.version += 1

        with self.assertRaises(LengthOrHashMismatchError):
            self._updater().refresh()

        self._assert_cache_equals(baseline)
        self._assert_no_transaction_left()

    # ------------------------------------------------------------------
    # Repeated crash/restart cycles: recovery must keep converging even when
    # interruptions happen more than once across different "process starts".
    # ------------------------------------------------------------------
    def test_repeated_crash_and_restart_cycles_converge(self) -> None:
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        real_replace = os.replace

        for _cycle in range(3):
            # Publish new metadata, then interrupt the refresh partway through
            # the commit phase.
            self.sim.targets.version += 1
            self.sim.update_snapshot()
            updater = self._updater()
            counter = {"n": 0}

            def failing_replace(
                src: str,
                dst: str,
                *args: object,
                counter: dict[str, int] = counter,
            ) -> None:
                if self._commit_destinations(self.metadata_dir, dst):
                    counter["n"] += 1
                    # Fail on the second committed rename of this cycle
                    if counter["n"] == 2:
                        raise OSError("simulated power loss")
                real_replace(src, dst, *args)

            with (
                patch("os.replace", side_effect=failing_replace),
                self.assertRaises(OSError),
            ):
                updater.refresh()

            # A brand new Updater models the next process start: its
            # constructor runs recovery and must reach a consistent state.
            restarted = self._updater()
            self._assert_no_transaction_left()

            # The recovered cache stays usable and internally coherent: the
            # root.json link resolves to an existing root_history file.
            link = os.path.join(self.metadata_dir, "root.json")
            self.assertTrue(os.path.exists(link))

            # A further refresh against the same repository succeeds and
            # leaves a complete, consistent top-level role set.
            restarted.refresh()
            state = self._baseline_snapshot()
            for role in (
                "root.json",
                "timestamp.json",
                "snapshot.json",
                "targets.json",
            ):
                self.assertIn(role, state)
            self._assert_no_transaction_left()

    # ------------------------------------------------------------------
    # Form 12: the same role is never applied inconsistently twice
    # ------------------------------------------------------------------
    def test_repeated_persist_is_idempotent(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"v1")
            txn.add_or_replace("timestamp.json", b"v1")
            txn.commit()
        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"v1")
        self._assert_no_transaction_left()


class RecoveryTest(unittest.TestCase):
    """Direct tests for journal-based crash recovery and re-entrancy."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        os.makedirs(self.metadata_dir)
        self.state_dir = os.path.join(self.metadata_dir, STATE_DIRNAME)
        os.makedirs(os.path.join(self.state_dir, "tmp"))
        self.cache = MetadataCache(self.metadata_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, rel: str, data: bytes) -> str:
        path = os.path.join(self.metadata_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _make_txn_dir(self, txn_id: str) -> tuple[str, str, str]:
        txn_dir = os.path.join(self.state_dir, f"txn-{txn_id}")
        staged = os.path.join(txn_dir, STAGED_DIRNAME)
        backup = os.path.join(txn_dir, BACKUP_DIRNAME)
        os.makedirs(staged, exist_ok=True)
        os.makedirs(backup, exist_ok=True)
        return txn_dir, staged, backup

    def _write_journal(self, txn_dir: str, state: str, entries: list) -> None:
        with open(os.path.join(txn_dir, JOURNAL_FILENAME), "w") as f:
            json.dump({"state": state, "entries": entries}, f)

    def test_prepared_transaction_is_rolled_back(self) -> None:
        self._write("timestamp.json", b"old-timestamp")
        self._write("targets.json", b"old-targets")
        txn_dir, staged, backup = self._make_txn_dir("abc")
        # replace timestamp: backup holds old content
        with open(os.path.join(backup, "1"), "wb") as f:
            f.write(b"old-timestamp")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"new-timestamp")
        # add a brand new file
        with open(os.path.join(staged, "2"), "wb") as f:
            f.write(b"new-snapshot")
        # Simulate that the first replace had happened before the crash
        os.replace(
            os.path.join(staged, "1"),
            os.path.join(self.metadata_dir, "timestamp.json"),
        )
        self._write("snapshot.json", b"new-snapshot")
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "replace",
                    "path": "timestamp.json",
                    "staged": "1",
                    "backup": "1",
                },
                {"op": "add", "path": "snapshot.json", "staged": "2"},
            ],
        )

        MetadataCache(self.metadata_dir).recover()

        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"old-timestamp")
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "snapshot.json"))
        )
        with open(os.path.join(self.metadata_dir, "targets.json"), "rb") as f:
            self.assertEqual(f.read(), b"old-targets")
        self.assertFalse(os.path.exists(txn_dir))

    def test_committed_transaction_is_rolled_forward(self) -> None:
        txn_dir, staged, _backup = self._make_txn_dir("def")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"committed-timestamp")
        self._write_journal(
            txn_dir,
            "committed",
            [{"op": "add", "path": "timestamp.json", "staged": "1"}],
        )

        MetadataCache(self.metadata_dir).recover()

        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"committed-timestamp")
        self.assertFalse(os.path.exists(txn_dir))

    def test_recovery_is_reentrant_after_interruption(self) -> None:
        self._write("timestamp.json", b"old")
        txn_dir, staged, backup = self._make_txn_dir("ghi")
        with open(os.path.join(backup, "1"), "wb") as f:
            f.write(b"old")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"new")
        os.replace(
            os.path.join(staged, "1"),
            os.path.join(self.metadata_dir, "timestamp.json"),
        )
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "replace",
                    "path": "timestamp.json",
                    "staged": "1",
                    "backup": "1",
                }
            ],
        )

        cache = MetadataCache(self.metadata_dir)
        # Simulate the first recovery attempt being interrupted before it
        # finishes: run only the rollback step, do not remove the txn dir.
        with open(os.path.join(txn_dir, JOURNAL_FILENAME)) as f:
            journal = json.load(f)
        cache._roll_back(journal["entries"], backup)

        # A second, full recovery (as on the next process start) must still
        # converge to the same consistent state.
        MetadataCache(self.metadata_dir).recover()

        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"old")
        self.assertFalse(os.path.exists(txn_dir))
        # Running recovery yet again changes nothing
        MetadataCache(self.metadata_dir).recover()
        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"old")

    def test_leftover_temp_files_are_cleaned_on_startup(self) -> None:
        tmp = os.path.join(self.state_dir, "tmp")
        os.makedirs(tmp, exist_ok=True)
        leftover = os.path.join(tmp, "tmpold123")
        with open(leftover, "wb") as f:
            f.write(b"partial")

        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(leftover))
        self.assertTrue(os.path.isdir(tmp))

    def test_journalless_transaction_dir_is_discarded(self) -> None:
        txn_dir, _staged, _backup = self._make_txn_dir("orphan")
        self.assertTrue(os.path.isdir(txn_dir))

        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(txn_dir))

    def test_root_link_rollback_restores_previous_target(self) -> None:
        os.makedirs(os.path.join(self.metadata_dir, "root_history"))
        self._write("root_history/1.root.json", b"root1")
        os.symlink(
            os.path.join("root_history", "1.root.json"),
            os.path.join(self.metadata_dir, "root.json"),
        )
        txn_dir, staged, _backup = self._make_txn_dir("link")
        self._write("root_history/2.root.json", b"root2-staged-original")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"root2")
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "add",
                    "path": "root_history/2.root.json",
                    "staged": "1",
                },
                {
                    "op": "link",
                    "path": "root.json",
                    "target": "root_history/2.root.json",
                    "old_target": "root_history/1.root.json",
                },
            ],
        )
        # Apply the link switch as if commit had partially run
        link = os.path.join(self.metadata_dir, "root.json")
        os.remove(link)
        os.symlink("root_history/2.root.json", link)

        MetadataCache(self.metadata_dir).recover()

        self.assertEqual(
            os.readlink(link),
            os.path.join("root_history", "1.root.json"),
        )
        # The add (root v2, no backup) is undone as well
        self.assertFalse(
            os.path.exists(
                os.path.join(self.metadata_dir, "root_history", "2.root.json")
            )
        )

    def test_transaction_context_aborts_without_commit(self) -> None:
        self._write("timestamp.json", b"old")
        cache = MetadataCache(self.metadata_dir)
        with (
            self.assertRaises(RuntimeError),
            cache.transaction() as txn,
        ):
            txn.add_or_replace("timestamp.json", b"new")
            raise RuntimeError("boom")

        with open(os.path.join(self.metadata_dir, "timestamp.json"), "rb") as f:
            self.assertEqual(f.read(), b"old")
        self._assert_no_txn_dirs(cache)

    def test_paths_outside_cache_are_rejected(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with (
            cache.transaction() as txn,
            self.assertRaises(StorageError),
        ):
            txn.add_or_replace("../../escape.json", b"x")

    # ------------------------------------------------------------------
    # Additional branch coverage for the transaction state machine and the
    # recovery edge cases.
    # ------------------------------------------------------------------
    def test_recover_without_state_dir_is_a_noop(self) -> None:
        # Remove the state directory entirely: recovery must simply return.
        shutil.rmtree(self.state_dir)
        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(self.state_dir))

    def test_recover_removes_leftover_tmp_subdirectory(self) -> None:
        tmp = os.path.join(self.state_dir, "tmp")
        os.makedirs(os.path.join(tmp, "stale-txn"))
        with open(os.path.join(tmp, "stale-txn", "part"), "wb") as f:
            f.write(b"x")
        MetadataCache(self.metadata_dir).recover()
        # The tmp directory is recreated empty, its contents removed
        self.assertEqual(os.listdir(tmp), [])

    def test_recover_ignores_non_transaction_entries(self) -> None:
        # A regular file that is not a txn directory must be left untouched
        marker = os.path.join(self.state_dir, "marker")
        with open(marker, "wb") as f:
            f.write(b"keep")
        MetadataCache(self.metadata_dir).recover()
        self.assertTrue(os.path.exists(marker))

    def test_recover_corrupt_journal_is_discarded(self) -> None:
        txn_dir, _staged, _backup = self._make_txn_dir("corrupt")
        with open(os.path.join(txn_dir, JOURNAL_FILENAME), "w") as f:
            f.write("{not valid json")
        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(txn_dir))

    def test_recover_journal_missing_entries_key_is_discarded(self) -> None:
        txn_dir, _staged, _backup = self._make_txn_dir("nokeys")
        with open(os.path.join(txn_dir, JOURNAL_FILENAME), "w") as f:
            json.dump({"state": "prepared"}, f)
        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(txn_dir))

    def test_recover_failure_raises_storage_error(self) -> None:
        self._write("timestamp.json", b"old")
        txn_dir, staged, backup = self._make_txn_dir("broken")
        with open(os.path.join(backup, "1"), "wb") as f:
            f.write(b"old")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"new")
        # The replacement already happened before the crash...
        os.replace(
            os.path.join(staged, "1"),
            os.path.join(self.metadata_dir, "timestamp.json"),
        )
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "replace",
                    "path": "timestamp.json",
                    "staged": "1",
                    "backup": "1",
                }
            ],
        )
        # ...and now restoring the backup fails: recovery must surface a
        # StorageError instead of silently continuing with a half-state.
        real_replace = os.replace

        def fail_restore(src: str, dst: str, *args: object) -> None:
            if dst.endswith("timestamp.json"):
                raise OSError("cannot restore")
            real_replace(src, dst, *args)

        with (
            patch("os.replace", side_effect=fail_restore),
            self.assertRaises(StorageError),
        ):
            MetadataCache(self.metadata_dir).recover()

    def test_roll_forward_is_idempotent_when_already_applied(self) -> None:
        txn_dir, staged, _backup = self._make_txn_dir("forward")
        with open(os.path.join(staged, "1"), "wb") as f:
            f.write(b"committed")
        self._write_journal(
            txn_dir,
            "committed",
            [{"op": "add", "path": "timestamp.json", "staged": "1"}],
        )
        cache = MetadataCache(self.metadata_dir)
        cache.recover()
        first = self._read("timestamp.json")
        # Run again: the staged file is gone but recovery must still succeed
        cache.recover()
        self.assertEqual(self._read("timestamp.json"), first)

    def test_roll_forward_skips_link_already_correct(self) -> None:
        os.makedirs(os.path.join(self.metadata_dir, "root_history"))
        self._write("root_history/1.root.json", b"root1")
        os.symlink(
            os.path.join("root_history", "1.root.json"),
            os.path.join(self.metadata_dir, "root.json"),
        )
        txn_dir, _staged, _backup = self._make_txn_dir("linkforward")
        self._write_journal(
            txn_dir,
            "committed",
            [
                {
                    "op": "link",
                    "path": "root.json",
                    "target": "root_history/1.root.json",
                    "old_target": None,
                }
            ],
        )
        # Link already points at the journal target: no error, txn removed
        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(os.path.exists(txn_dir))

    def test_roll_back_replace_with_missing_backup_keeps_file(self) -> None:
        self._write("timestamp.json", b"current")
        txn_dir, _staged, _backup = self._make_txn_dir("nobackup")
        # Journal references a backup that does not exist: the rename had not
        # happened, so recovery must leave the committed file untouched.
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "replace",
                    "path": "timestamp.json",
                    "staged": "1",
                    "backup": "missing",
                }
            ],
        )
        MetadataCache(self.metadata_dir).recover()
        self.assertEqual(self._read("timestamp.json"), b"current")
        self.assertFalse(os.path.exists(txn_dir))

    def test_roll_back_link_with_null_old_target_removes_link(self) -> None:
        os.symlink(
            "root_history/1.root.json",
            os.path.join(self.metadata_dir, "root.json"),
        )
        txn_dir, _staged, _backup = self._make_txn_dir("nulllink")
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "link",
                    "path": "root.json",
                    "target": "root_history/2.root.json",
                    "old_target": None,
                }
            ],
        )
        MetadataCache(self.metadata_dir).recover()
        self.assertFalse(
            os.path.lexists(os.path.join(self.metadata_dir, "root.json"))
        )

    def test_roll_back_hardlink_style_backup_target(self) -> None:
        # A link entry recorded with a backup: marker represents the Windows
        # hard-link fallback where the old content was copied to backup.
        self._write("root.json", b"old-root")
        txn_dir, _staged, backup = self._make_txn_dir("hardlink")
        with open(os.path.join(backup, "7"), "wb") as f:
            f.write(b"old-root")
        self._write_journal(
            txn_dir,
            "prepared",
            [
                {
                    "op": "link",
                    "path": "root.json",
                    "target": "root_history/2.root.json",
                    "old_target": "backup:7",
                }
            ],
        )
        MetadataCache(self.metadata_dir).recover()
        self.assertEqual(self._read("root.json"), b"old-root")

    def test_commit_then_close_is_idempotent(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"data")
            txn.commit()
        self.assertTrue(txn.is_committed)
        self.assertTrue(txn.is_closed)
        # Closing again must be a no-op rather than raise
        txn.close()

    def test_close_without_commit_rolls_back(self) -> None:
        self._write("timestamp.json", b"old")
        cache = MetadataCache(self.metadata_dir)
        txn_cm = cache.transaction()
        txn = txn_cm.__enter__()
        txn.add_or_replace("timestamp.json", b"new")
        # Releasing via close() without commit must undo the staging
        txn.close()
        self.assertTrue(txn.is_closed)
        self.assertEqual(self._read("timestamp.json"), b"old")
        # The generator context manager still owns the lock: exit it cleanly
        txn_cm.__exit__(None, None, None)

    def test_commit_on_closed_transaction_raises(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"data")
        # Exiting without committing rolled back and closed the transaction
        with self.assertRaises(StorageError):
            txn.commit()

    def test_add_or_replace_backs_up_existing_file(self) -> None:
        self._write("timestamp.json", b"old")
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"new")
            # A backup of the pre-existing content was captured
            backups = os.listdir(
                os.path.join(self.state_dir, f"txn-{txn.id}", BACKUP_DIRNAME)
            )
            self.assertEqual(backups, ["1"])
            txn.commit()
        self.assertEqual(self._read("timestamp.json"), b"new")

    def test_nonblocking_lock_succeeds_when_free(self) -> None:
        # A short timeout on an uncontended lock must acquire immediately.
        cache = MetadataCache(self.metadata_dir, lock_timeout=1.0)
        with cache.transaction() as txn:
            self.assertFalse(txn.is_committed)
            txn.commit()

    def test_journal_write_failure_cleans_temp_file(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"data")
            with (
                patch("os.fsync", side_effect=OSError("io error")),
                self.assertRaises(OSError),
            ):
                # The journal temp file is removed when journal writing fails;
                # the staging data is rolled back and nothing is committed.
                txn.commit()
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "timestamp.json"))
        )

    def test_commit_is_idempotent_when_called_twice(self) -> None:
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.add_or_replace("timestamp.json", b"data")
            txn.commit()
            # A second commit on an already committed txn is a harmless no-op
            txn.commit()
        self.assertEqual(self._read("timestamp.json"), b"data")

    def test_symlink_failure_other_than_windows_privilege_propagates(
        self,
    ) -> None:
        # An OSError without the Windows privilege winerror must surface,
        # not be swallowed by the hard-link fallback.
        cache = MetadataCache(self.metadata_dir)
        err = OSError("not supported")
        with (
            patch("os.symlink", side_effect=err),
            self.assertRaises(OSError),
            cache.transaction() as txn,
        ):
            txn.add_or_replace(
                os.path.join("root_history", "1.root.json"), b"root"
            )
            txn.update_root_link(os.path.join("root_history", "1.root.json"))
            txn.commit()

    def test_recover_creates_tmp_dir_when_missing(self) -> None:
        shutil.rmtree(os.path.join(self.state_dir, "tmp"))
        MetadataCache(self.metadata_dir).recover()
        self.assertTrue(os.path.isdir(os.path.join(self.state_dir, "tmp")))

    def test_recover_skips_non_directory_transaction_entry(self) -> None:
        # A "txn-" name that is a file rather than a directory is skipped
        path = os.path.join(self.state_dir, "txn-not-a-dir")
        with open(path, "wb") as f:
            f.write(b"x")
        # Must not raise; the file is simply ignored
        MetadataCache(self.metadata_dir).recover()
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_staging_failure_removes_temp_file(self) -> None:
        # When moving the staged file into the transaction directory fails,
        # the temporary file created in tmp/ must be removed.
        cache = MetadataCache(self.metadata_dir)
        real_replace = os.replace

        def fail_staged_move(src: str, dst: str, *args: object) -> None:
            if STAGED_DIRNAME in dst:
                raise OSError("staging failed")
            real_replace(src, dst, *args)

        with (
            patch("os.replace", side_effect=fail_staged_move),
            self.assertRaises(OSError),
            cache.transaction() as txn,
        ):
            txn.add_or_replace("timestamp.json", b"data")

        # Nothing was committed and no staged temp file remains
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "timestamp.json"))
        )
        tmp = os.path.join(self.state_dir, "tmp")
        self.assertEqual(os.listdir(tmp), [])

    def test_root_link_backs_up_plain_file_target(self) -> None:
        # When root.json is a plain file (Windows hard-link fallback case)
        # rather than a symlink, updating the link backs its content up.
        self._write("root.json", b"old-root")
        cache = MetadataCache(self.metadata_dir)
        with cache.transaction() as txn:
            txn.update_root_link(os.path.join("root_history", "2.root.json"))
            link_entries = [e for e in txn._entries if e["op"] == "link"]
            self.assertEqual(len(link_entries), 1)
            self.assertTrue(link_entries[0]["old_target"].startswith("backup:"))
            backups = os.listdir(
                os.path.join(self.state_dir, f"txn-{txn.id}", BACKUP_DIRNAME)
            )
            self.assertIn(
                link_entries[0]["old_target"].split(":", 1)[1], backups
            )
            txn.rollback()

    def _read(self, rel: str) -> bytes:
        with open(os.path.join(self.metadata_dir, rel), "rb") as f:
            return f.read()

    @staticmethod
    def _assert_no_txn_dirs(cache: MetadataCache) -> None:
        for name in os.listdir(cache.state_dir):
            assert not name.startswith("txn-")


if __name__ == "__main__":
    unittest.main()
