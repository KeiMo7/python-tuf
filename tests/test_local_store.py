# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for the multi-process safe local metadata store and Updater cache.

These tests cover the failure modes that a plain set of role-named files cannot
handle: concurrent writers, a process killed at an arbitrary point, and readers
that must never observe a mixture of old and new metadata.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import utils
from tests.file_repository import FileFetcher, dump_repository
from tests.repository_simulator import RepositorySimulator
from tuf.api import exceptions
from tuf.api.metadata import Metadata, Snapshot, Targets, Timestamp
from tuf.ngclient import Updater, UpdaterConfig
from tuf.ngclient._internal import local_store
from tuf.ngclient._internal.local_store import MetadataStore

TESTS_DIR = Path(__file__).resolve().parent


class _StoreTestCase(unittest.TestCase):
    """Common setup: client dirs plus a dumped file-based repository."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        self.repo_dir = os.path.join(self.temp_dir.name, "repo")
        os.mkdir(self.metadata_dir)
        os.mkdir(self.targets_dir)
        self.sim = RepositorySimulator()
        self.publish()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def publish(self) -> None:
        """Dump the current simulator state so child processes can fetch it."""
        dump_repository(self.sim, self.repo_dir)

    def updater(self, **config: object) -> Updater:
        root_path = os.path.join(self.metadata_dir, "root.json")
        if os.path.exists(root_path):
            with open(root_path, "rb") as f:
                bootstrap = f.read()
        else:
            with open(
                os.path.join(self.repo_dir, "metadata", "1.root.json"), "rb"
            ) as f:
                bootstrap = f.read()
        return Updater(
            self.metadata_dir,
            f"file://{self.repo_dir}/metadata/",
            self.targets_dir,
            f"file://{self.repo_dir}/targets/",
            FileFetcher(),
            config=UpdaterConfig(**config),  # type: ignore[arg-type]
            bootstrap=bootstrap,
        )

    def child_env(self, **extra: str) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": str(TESTS_DIR.parent),
            "TUF_CRASH_METADATA_DIR": self.metadata_dir,
            "TUF_CRASH_TARGETS_DIR": self.targets_dir,
            "TUF_CRASH_REPO_DIR": self.repo_dir,
            **extra,
        }

    def run_child(self, hook: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TESTS_DIR / "_crash_helper.py"), hook],
            env=self.child_env(**env),
            capture_output=True,
            timeout=120,
            check=False,
        )

    def spawn_child(self, hook: str, **env: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(TESTS_DIR / "_crash_helper.py"), hook],
            env=self.child_env(**env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class TestStoreLayout(_StoreTestCase):
    """Basic store behaviour: generations, atomicity and recovery."""

    def test_refresh_commits_one_generation(self) -> None:
        self.updater().refresh()

        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self.assertEqual(
            generation.versions,
            {"root": 1, "timestamp": 1, "snapshot": 1, "targets": 1},
        )
        # exactly one committed generation, no staging leftovers
        self.assertEqual(
            os.listdir(os.path.join(self.metadata_dir, "generations")),
            ["gen-0000000001"],
        )
        for name in os.listdir(self.metadata_dir):
            self.assertFalse(name.startswith((".current-", ".root-")), name)

    def test_new_generation_keeps_previous_immutable(self) -> None:
        self.updater().refresh()
        first = MetadataStore(self.metadata_dir).verify()
        assert first is not None

        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        self.updater().refresh()
        second = MetadataStore(self.metadata_dir).verify()
        assert second is not None

        self.assertEqual(second.number, first.number + 1)
        self.assertEqual(second.versions["targets"], 2)
        # the previous generation is untouched
        self.assertEqual(first.versions["targets"], 1)
        # carried-over metadata is identical, updated metadata is not
        self.assertEqual(first.read("root"), second.read("root"))
        self.assertNotEqual(first.read("targets"), second.read("targets"))

    def test_damaged_generation_rolls_back_and_reports(self) -> None:
        self.updater().refresh()
        first = MetadataStore(self.metadata_dir).verify()
        assert first is not None

        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        self.updater().refresh()
        second = MetadataStore(self.metadata_dir).verify()
        assert second is not None
        self.assertEqual(second.versions["targets"], 2)

        # simulate damage: truncate the newest generation's targets file
        with open(os.path.join(second.path, "targets.json"), "wb") as f:
            f.write(b"{}")

        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ) as logs:
            recovered = MetadataStore(self.metadata_dir).verify()

        # The rollback is reported and the previous complete generation is used
        self.assertTrue(
            any("rolling back" in line for line in logs.output), logs.output
        )
        assert recovered is not None
        self.assertEqual(recovered.number, first.number)
        self.assertEqual(recovered.versions["targets"], 1)

        # the rollback is persisted: a new store sees the previous generation
        again = MetadataStore(self.metadata_dir).verify()
        assert again is not None
        self.assertEqual(again.number, first.number)

    def test_no_complete_generation_raises(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        os.remove(os.path.join(generation.path, "manifest.json"))

        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_missing_file_is_detected(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        os.remove(os.path.join(generation.path, "targets.json"))

        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_lock_timeout_is_explicit(self) -> None:
        store = MetadataStore(
            self.metadata_dir, lock_timeout=0.05, lease_timeout=None
        )
        other = MetadataStore(self.metadata_dir, lock_timeout=0.0)

        with (
            store.lock(),
            self.assertRaises(exceptions.LockTimeoutError) as ctx,
        ):
            other.lock().__enter__()
        self.assertIn("lock", str(ctx.exception))

        # the failed acquisition does not leave the lock unusable
        with other.lock():
            pass

    def test_reentrant_lock(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with store.lock():
            with store.lock():
                self.assertIsNotNone(store._lock_fd)
            self.assertIsNotNone(store._lock_fd)
        self.assertIsNone(store._lock_fd)

    def test_reader_never_needs_to_repair(self) -> None:
        """A reader always receives a complete set from one generation."""
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        generation = store.load()
        assert generation is not None
        # a second read resolves the same cached generation
        self.assertIs(store.load(), generation)
        for role in generation.roles:
            generation.read(role)


class TestCrashRecovery(_StoreTestCase):
    """A process killed at an arbitrary point must leave a readable, monotonic
    local state."""

    def _verify_readable(self) -> MetadataStore:
        store = MetadataStore(self.metadata_dir)
        generation = store.verify()
        assert generation is not None
        for role in generation.roles:
            self.assertTrue(generation.read(role))
        return store

    def test_crash_during_staging(self) -> None:
        result = self.run_child("staging")
        self.assertEqual(result.returncode, 17, result.stderr)

        # No committed generation is visible and nothing is corrupted
        self.assertIsNone(MetadataStore(self.metadata_dir).verify())

        # a fresh refresh succeeds and cleans the staging directory
        self.updater().refresh()
        self._verify_readable()
        generations = os.listdir(os.path.join(self.metadata_dir, "generations"))
        self.assertEqual(generations, ["gen-0000000001"])

    def test_crash_before_pointer_commit_adopts_newer_generation(self) -> None:
        # a complete refresh first, so a previous generation exists
        self.updater().refresh()
        before = MetadataStore(self.metadata_dir).verify()
        assert before is not None

        # repository moves forward; the child dies after renaming its
        # generation but before replacing the pointer
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        result = self.run_child("renamed")
        self.assertEqual(result.returncode, 18, result.stderr)

        # the pointer still points at the old generation
        with open(os.path.join(self.metadata_dir, "current")) as f:
            self.assertEqual(json.load(f)["generation"], before.number)

        # recovery adopts the complete newer generation
        recovered = MetadataStore(self.metadata_dir).verify()
        assert recovered is not None
        self.assertEqual(recovered.number, before.number + 1)
        self.assertEqual(recovered.versions["targets"], 2)

        # a following refresh agrees with what was recovered
        updater = self.updater()
        updater.refresh()
        self.assertEqual(updater.local_metadata_versions()["targets"], 2)

    def test_crash_after_pointer_commit(self) -> None:
        self.updater().refresh()
        before = MetadataStore(self.metadata_dir).verify()
        assert before is not None

        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        result = self.run_child("committed")
        self.assertEqual(result.returncode, 19, result.stderr)

        # the commit is complete: pointer and generation agree
        after = MetadataStore(self.metadata_dir).verify()
        assert after is not None
        self.assertEqual(after.number, before.number + 1)
        self.assertEqual(after.versions["targets"], 2)

    def test_versions_are_monotonic_in_each_crash_case(self) -> None:
        for hook in ("staging", "renamed", "committed"):
            with self.subTest(hook=hook):
                metadata_dir = os.path.join(self.temp_dir.name, hook)
                targets_dir = os.path.join(self.temp_dir.name, f"{hook}-t")
                os.mkdir(metadata_dir)
                os.mkdir(targets_dir)
                self.metadata_dir = metadata_dir
                self.targets_dir = targets_dir

                self.updater().refresh()
                previous = MetadataStore(metadata_dir).verify()
                assert previous is not None

                self.sim.targets.version += 1
                self.sim.update_snapshot()
                self.publish()
                self.run_child(hook)

                current = MetadataStore(metadata_dir).verify()
                assert current is not None
                for role, version in previous.versions.items():
                    self.assertGreaterEqual(
                        current.versions[role], version, (hook, role)
                    )
                self.assertGreaterEqual(current.versions["targets"], 1)

    def test_crash_mid_target_download_does_not_publish_target(self) -> None:
        self.sim.add_target("targets", b"content", "target")
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()

        result = self.run_child("target")
        self.assertNotEqual(result.returncode, 0)

        updater = self.updater()
        updater.refresh()
        info = updater.get_targetinfo("target")
        assert info is not None
        # no partially written target is visible
        self.assertIsNone(updater.find_cached_target(info))
        self.assertEqual(
            [
                name
                for name in os.listdir(self.targets_dir)
                if not name.startswith(".download-")
            ],
            [],
        )
        # and a later download succeeds
        updater.download_target(info)
        self.assertIsNotNone(updater.find_cached_target(info))

    def test_old_generations_are_pruned_current_is_kept(self) -> None:
        for _ in range(4):
            self.updater(generations_to_keep=1).refresh()
            self.sim.targets.version += 1
            self.sim.update_snapshot()
            self.publish()

        generations = sorted(
            os.listdir(os.path.join(self.metadata_dir, "generations"))
        )
        # current + one older generation are kept
        self.assertEqual(len(generations), 2)
        self.assertEqual(generations[-1], "gen-0000000004")


class TestMultiProcess(_StoreTestCase):
    """Concurrent refreshes of one metadata directory end in one complete
    generation, and every process observes the same versions."""

    def _wait_all(
        self, processes: list[subprocess.Popen]
    ) -> list[dict[str, object]]:
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            self.assertEqual(
                process.returncode,
                0,
                f"worker failed: {stderr.decode(errors='replace')}",
            )
            results.append(json.loads(stdout.decode()))
        return results

    def test_concurrent_first_refresh(self) -> None:
        processes = [self.spawn_child("refresh") for _ in range(4)]
        results = self._wait_all(processes)

        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        for result in results:
            self.assertEqual(result["versions"], generation.versions)
        # the visible generation is the newest one and is complete
        self.assertEqual(
            sorted(os.listdir(os.path.join(self.metadata_dir, "generations")))[
                -1
            ],
            f"gen-{generation.number:010d}",
        )
        self.assertEqual(
            [
                name
                for name in os.listdir(self.metadata_dir)
                if name.startswith(".current-")
            ],
            [],
        )

    def _read_and_check(self, errors: list[str], seen: list[int]) -> None:
        """Read the visible generation and assert it is internally consistent."""
        try:
            generation = MetadataStore(self.metadata_dir).verify()
        except exceptions.LocalMetadataError as e:
            errors.append(str(e))
            return
        if generation is None:
            return
        timestamp = (
            Metadata[Timestamp].from_bytes(generation.read("timestamp")).signed
        )
        snapshot = (
            Metadata[Snapshot].from_bytes(generation.read("snapshot")).signed
        )
        targets = (
            Metadata[Targets].from_bytes(generation.read("targets")).signed
        )
        # mixed generations are exactly what this asserts against: the files
        # in one generation must reference each other
        if timestamp.snapshot_meta.version != snapshot.version:
            errors.append(
                f"timestamp v{timestamp.version} points at snapshot "
                f"v{timestamp.snapshot_meta.version}, generation has "
                f"v{snapshot.version}"
            )
        if snapshot.meta["targets.json"].version != targets.version:
            errors.append(
                f"snapshot v{snapshot.version} points at targets "
                f"v{snapshot.meta['targets.json'].version}, generation "
                f"has v{targets.version}"
            )
        if generation.versions["targets"] != targets.version:
            errors.append("manifest version does not match file")
        seen.append(generation.number)

    def test_concurrent_refresh_with_concurrent_readers(self) -> None:
        """Readers running during commits always see one self-consistent set."""
        for _ in range(5):
            self.updater().refresh()
            self.sim.targets.version += 1
            self.sim.update_snapshot()
            self.publish()

        stop = threading.Event()
        errors: list[str] = []
        seen: list[int] = []

        def reader() -> None:
            while not stop.is_set():
                self._read_and_check(errors, seen)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for thread in threads:
            thread.start()
        try:
            # the fetch from a local file server is fast: keep committing so
            # the readers have something to observe
            for _ in range(5):
                processes = [self.spawn_child("refresh") for _ in range(4)]
                self._wait_all(processes)
                self.sim.targets.version += 1
                self.sim.update_snapshot()
                self.publish()
                deadline = time.monotonic() + 0.05
                while time.monotonic() < deadline:
                    time.sleep(0.005)
        finally:
            stop.set()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertTrue(seen)

    def test_lock_is_released_when_holder_dies(self) -> None:
        # a child takes the lock and is killed without releasing it
        child = self.spawn_child("lock")
        assert child.stdout is not None
        line = child.stdout.readline()
        self.assertIn(b"locked", line)
        child.kill()
        child.wait()

        # the OS released the lock: a new process can take it immediately
        store = MetadataStore(self.metadata_dir, lock_timeout=5.0)
        start = time.monotonic()
        with store.lock():
            pass
        self.assertLess(time.monotonic() - start, 2.0)

    def test_contending_writer_fails_explicitly(self) -> None:
        """A writer that cannot get the lock fails: it does not write anyway."""
        child = self.spawn_child("lock")
        assert child.stdout is not None
        self.assertIn(b"locked", child.stdout.readline())
        try:
            store = MetadataStore(self.metadata_dir, lock_timeout=0.1)
            with self.assertRaises(exceptions.LockTimeoutError):
                store.lock().__enter__()
        finally:
            child.kill()
            child.wait()

        # now the lock is free again and refresh succeeds
        self.updater().refresh()
        self.assertIsNotNone(MetadataStore(self.metadata_dir).verify())

    def test_unverified_target_is_never_cached(self) -> None:
        self.sim.add_target("targets", b"content", "target")
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()

        updater = self.updater()
        updater.refresh()
        info = updater.get_targetinfo("target")
        assert info is not None

        # repository serves different bytes for the same target
        self.sim.target_files["target"].data = b"BAD"
        self.publish()

        with self.assertRaises(exceptions.LengthOrHashMismatchError):
            updater.download_target(info)

        # nothing was published, not even a temporary file
        self.assertEqual(os.listdir(self.targets_dir), [])
        self.assertIsNone(updater.find_cached_target(info))

        # a correct download afterwards succeeds
        self.sim.target_files["target"].data = b"content"
        self.publish()
        updater.download_target(info)
        self.assertIsNotNone(updater.find_cached_target(info))

    def test_cached_target_is_reverified(self) -> None:
        self.sim.add_target("targets", b"content", "target")
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()

        updater = self.updater()
        updater.refresh()
        info = updater.get_targetinfo("target")
        assert info is not None
        path = updater.download_target(info)

        # corrupt the cached file: it must not be reported as a cache hit
        with open(path, "wb") as f:
            f.write(b"corrupt")
        self.assertIsNone(updater.find_cached_target(info))


class TestStoreEdgeCases(_StoreTestCase):
    """Corrupted on-disk state, lock diagnostics and configuration."""

    def test_corrupt_pointer_is_ignored(self) -> None:
        self.updater().refresh()
        pointer_path = os.path.join(self.metadata_dir, "current")
        with open(pointer_path, "wb") as f:
            f.write(b"not json")

        # The pointer is unreadable: no generation is claimed to be visible.
        # The generation on disk is complete, so recovery adopts it.
        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ):
            generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self.assertEqual(generation.versions["targets"], 1)

    def test_pointer_to_unknown_generation_falls_back(self) -> None:
        """A pointer to a missing generation never hides a complete one."""
        self.updater().refresh()
        pointer_path = os.path.join(self.metadata_dir, "current")
        with open(pointer_path) as f:
            pointer = json.load(f)
        pointer["generation"] += 100
        with open(pointer_path, "w") as f:
            json.dump(pointer, f)

        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ):
            generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self.assertEqual(generation.number, 1)

    def test_no_complete_generation_for_corrupt_pointer(self) -> None:
        """A damaged pointer with no complete generation is reported."""
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        os.remove(os.path.join(generation.path, "manifest.json"))
        pointer_path = os.path.join(self.metadata_dir, "current")
        with open(pointer_path) as f:
            pointer = json.load(f)
        pointer["generation"] = 42
        with open(pointer_path, "w") as f:
            json.dump(pointer, f)

        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_pointers_with_wrong_manifest_digest_are_rejected(self) -> None:
        self.updater().refresh()
        pointer_path = os.path.join(self.metadata_dir, "current")
        with open(pointer_path) as f:
            pointer = json.load(f)
        pointer["manifest"] = "0" * 64
        with open(pointer_path, "w") as f:
            json.dump(pointer, f)

        # The pointer and the generation disagree, and there is no other
        # complete generation: nothing can be trusted.
        with (
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            MetadataStore(self.metadata_dir).verify()

    def test_manifest_format_is_validated(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        manifest_path = os.path.join(generation.path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["format"] = 999
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_lock_holder_info_reports_owner(self) -> None:
        store = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        other = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        with (
            store.lock(),
            self.assertRaises(exceptions.LockTimeoutError) as ctx,
        ):
            other.lock().__enter__()
        message = str(ctx.exception)
        self.assertIn("held by pid", message)
        self.assertIn(str(os.getpid()), message)
        self.assertIn("last heartbeat", message)

    def test_lease_reports_stuck_holder(self) -> None:
        store = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        other = MetadataStore(
            self.metadata_dir, lock_timeout=0.0, lease_timeout=0.0
        )
        with store.lock():
            # age the owner record so the lease looks expired
            with open(store._lock_path, "rb") as f:
                record = json.load(f)
            record["heartbeat"] = 0
            with open(store._lock_path, "w") as f:
                json.dump(record, f)
            with self.assertRaises(exceptions.LockTimeoutError) as ctx:
                other.lock().__enter__()
        self.assertIn("lease appears expired", str(ctx.exception))

    def test_owner_record_absent(self) -> None:
        store = MetadataStore(self.metadata_dir)
        self.assertEqual(store._holder_info(), "lock holder is unknown")

    def test_generation_numbers_ignore_unrelated_entries(self) -> None:
        self.updater().refresh()
        generations_dir = os.path.join(self.metadata_dir, "generations")
        os.mkdir(os.path.join(generations_dir, "not-a-generation"))
        os.mkdir(os.path.join(generations_dir, "gen-not-a-number"))
        store = MetadataStore(self.metadata_dir)
        self.assertEqual(store._generation_numbers(), [1])

    def test_config_options_reach_the_store(self) -> None:
        updater = self.updater(
            lock_timeout=1.5, lease_timeout=None, generations_to_keep=2
        )
        self.assertEqual(updater._store._lock_timeout, 1.5)
        self.assertIsNone(updater._store._lease_timeout)
        self.assertEqual(updater._store._generations_to_keep, 2)

    def test_local_metadata_versions_without_generation(self) -> None:
        updater = self.updater()
        # No generation exists yet: the bootstrap root is reported
        self.assertEqual(updater.local_metadata_versions(), {"root": 1})

    def test_stale_metadata_is_not_committed(self) -> None:
        """A writer that lost a race never rolls a role back."""
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        base = store.load()
        assert base is not None
        # pretend this writer verified an older targets metadata
        stale = self.sim.fetch_metadata("targets")
        with store.lock():
            new_base = store.commit(base, {"targets": stale}, {"targets": 0})
        # nothing was written, the existing generation is returned
        self.assertIs(new_base, base)

    def test_stale_role_is_dropped_but_others_committed(self) -> None:
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        base = store.load()
        assert base is not None
        targets = self.sim.fetch_metadata("targets")
        timestamp = self.sim.fetch_metadata("timestamp")
        with store.lock():
            new_base = store.commit(
                base,
                {"targets": targets, "timestamp": timestamp},
                {"targets": 0, "timestamp": 5},
            )
        self.assertEqual(new_base.versions["targets"], 1)
        self.assertEqual(new_base.versions["timestamp"], 5)

    def test_rollback_allowed_explicitly(self) -> None:
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        base = store.load()
        assert base is not None
        targets = self.sim.fetch_metadata("targets")
        with store.lock():
            new_base = store.commit(
                base, {"targets": targets}, {"targets": 0}, allow_rollback=True
            )
        self.assertEqual(new_base.versions["targets"], 0)

    def test_commit_without_lock_raises(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with self.assertRaises(RuntimeError):
            store.commit(None, {}, {})

    def test_read_before_any_generation(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with self.assertRaises(FileNotFoundError):
            store.read("targets")
        self.assertIsNone(store.load())

    def test_read_missing_role_retries(self) -> None:
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        with self.assertRaises(FileNotFoundError):
            store.read("role1")

    def test_update_root_symlink_fallback_failure_is_raised(self) -> None:
        self.updater().refresh()
        updater = self.updater()
        with (
            patch.object(os, "symlink", side_effect=OSError("nope")),
            self.assertRaises(OSError),
        ):
            updater._replace_root_anchor(
                os.path.join(self.metadata_dir, "root.json"),
                os.path.join("generations", "gen-0000000001", "root.json"),
            )

    def test_lock_timeout_on_refresh_is_explicit(self) -> None:
        """A refresh that cannot get the lock fails; it never writes anyway."""
        updater = self.updater(lock_timeout=0.05)
        holder = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        with (
            holder.lock(),
            self.assertRaises(exceptions.LockTimeoutError),
        ):
            updater.refresh()

        # Nothing was committed by the failed refresh, and the next attempt
        # works once the lock is free.
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "current"))
        )
        self.updater(lock_timeout=0.05).refresh()
        self.assertIsNotNone(MetadataStore(self.metadata_dir).verify())

    def test_target_write_failure_leaves_no_file(self) -> None:
        self.sim.add_target("targets", b"content", "target")
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        updater = self.updater()
        updater.refresh()
        info = updater.get_targetinfo("target")
        assert info is not None

        with (
            patch.object(os, "replace", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            updater.download_target(info)

        # the temporary file is cleaned up, nothing visible is left behind
        self.assertEqual(os.listdir(self.targets_dir), [])
        self.assertIsNone(updater.find_cached_target(info))

    def test_recover_requires_lock(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with self.assertRaises(RuntimeError):
            store.recover()

    def test_commit_atomic_replace_failure_keeps_previous_visible(self) -> None:
        """A failed pointer write leaves the previous generation visible."""
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        base = store.load()
        assert base is not None
        targets = self.sim.fetch_metadata("targets")
        with (
            store.lock(),
            patch.object(os, "replace", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            store.commit(base, {"targets": targets}, {"targets": 9})
            # not reached: the commit raises
        self.assertFalse(
            any(
                name.startswith(".current-")
                for name in os.listdir(self.metadata_dir)
            )
        )
        # The failed commit did not change the visible pointer. The new
        # generation is complete on disk (it was renamed before the pointer
        # write failed), so the next reader adopts it, exactly as it would
        # after a crash at that point: no verified data is ever lost.
        with open(os.path.join(self.metadata_dir, "current")) as f:
            self.assertEqual(json.load(f)["generation"], base.number)
        again = MetadataStore(self.metadata_dir).load()
        assert again is not None
        self.assertEqual(again.versions["targets"], 9)

    def test_manifest_entries_are_validated(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        manifest_path = os.path.join(generation.path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["files"]["targets"]["length"] = "twelve"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_target_hash_is_checked_on_verify(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        targets_path = os.path.join(generation.path, "targets.json")
        with open(targets_path, "rb") as f:
            data = f.read()
        # same length, different bytes: only a hash check can catch this
        corrupted = data.replace(b'"version"', b'"VERSION"')
        self.assertEqual(len(corrupted), len(data))
        with open(targets_path, "wb") as f:
            f.write(corrupted)

        with (
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            MetadataStore(self.metadata_dir).verify()

    def _two_generations(self) -> tuple[int, int]:
        """Create two committed generations and return their numbers."""
        self.updater().refresh()
        first = MetadataStore(self.metadata_dir).verify()
        assert first is not None
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        self.updater().refresh()
        second = MetadataStore(self.metadata_dir).verify()
        assert second is not None
        return first.number, second.number

    def _damage_same_length(self, path: str) -> None:
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data.replace(b'"version"', b'"VERSION"'))

    def test_verify_detects_hash_damage_and_rolls_back(self) -> None:
        first, second = self._two_generations()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self._damage_same_length(os.path.join(generation.path, "targets.json"))

        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ):
            recovered = MetadataStore(self.metadata_dir).verify()
        assert recovered is not None
        self.assertEqual(recovered.number, first)
        self.assertLess(recovered.number, second)

    def test_verify_detects_manifest_digest_mismatch(self) -> None:
        """A generation whose manifest changed after load is rejected."""
        first, second = self._two_generations()
        store = MetadataStore(self.metadata_dir)
        visible = store.load()
        assert visible is not None
        self.assertEqual(visible.number, second)
        mismatched = local_store.Generation(
            visible.number, visible.path, visible.roles, "0" * 64
        )
        original = store._load_generation

        def load_generation(
            number: int, verify_hashes: bool
        ) -> local_store.Generation:
            if verify_hashes and number == visible.number:
                return mismatched
            return original(number, verify_hashes)

        with (
            patch.object(
                store, "_load_generation", side_effect=load_generation
            ),
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
        ):
            recovered = store.verify()
        assert recovered is not None
        self.assertEqual(recovered.number, first)

    def test_verify_with_hash_damage_and_no_fallback(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self._damage_same_length(os.path.join(generation.path, "targets.json"))
        with (
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            MetadataStore(self.metadata_dir).verify()

    def test_verify_without_any_generation(self) -> None:
        self.assertIsNone(MetadataStore(self.metadata_dir).verify())

    def test_stale_temp_cleanup_when_dir_missing(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with store.lock():
            # generations directory does not exist yet: nothing to clean
            store._remove_staging_dirs()

    def test_stale_temp_cleanup_when_dirs_vanish(self) -> None:
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        with store.lock():
            store._generations_dir = os.path.join(
                self.metadata_dir, "missing-generations"
            )
            store._remove_staging_dirs()
            real_listdir = os.listdir

            def listdir(path: str) -> list[str]:
                if path == self.metadata_dir:
                    raise FileNotFoundError(path)
                return real_listdir(path)

            with patch.object(local_store.os, "listdir", listdir):
                store._remove_staging_dirs()

    def test_manifest_mismatch_for_pointer_generation_is_discarded(
        self,
    ) -> None:
        """Recovery never adopts a generation that contradicts the pointer."""
        self.updater().refresh()
        pointer_path = os.path.join(self.metadata_dir, "current")
        with open(pointer_path) as f:
            pointer = json.load(f)
        generation_path = local_store.MetadataStore(
            self.metadata_dir
        )._generation_path(pointer["generation"])
        manifest_path = os.path.join(generation_path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["files"]["targets"]["version"] = 7
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        store = MetadataStore(self.metadata_dir)
        with (
            store.lock(),
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ) as logs,
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            store.recover()
        self.assertTrue(
            any("does not match the pointer" in line for line in logs.output)
        )

    def test_generation_versions_ignore_bad_manifest_values(self) -> None:
        generation = local_store.Generation(
            1,
            self.metadata_dir,
            {"targets": {"filename": "t.json", "length": 1}},
            "0" * 64,
        )
        self.assertEqual(generation.versions, {})

    def test_load_uses_visible_generation_when_lock_is_busy(self) -> None:
        self.updater().refresh()
        # create a newer complete generation that is not yet visible
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        MetadataStore(self.metadata_dir).verify()
        self.run_child("renamed")

        store = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        holder = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        with (
            holder.lock(),
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="WARNING"
            ),
        ):
            generation = store.load()
        # a valid generation is visible: never block, never return partial data
        assert generation is not None
        self.assertEqual(generation.versions["targets"], 1)

    def test_recover_adopts_new_generation(self) -> None:
        self.updater().refresh()
        first = MetadataStore(self.metadata_dir).verify()
        assert first is not None
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        self.run_child("renamed")

        store = MetadataStore(self.metadata_dir)
        with store.lock():
            recovered = store.recover()
        assert recovered is not None
        self.assertEqual(recovered.number, first.number + 1)

    def test_recover_without_pointer_and_no_generations(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with store.lock():
            self.assertIsNone(store.recover())

    def test_recover_with_pointer_but_no_complete_generation(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        self._damage_same_length(os.path.join(generation.path, "targets.json"))
        store = MetadataStore(self.metadata_dir)
        with (
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
            store.lock(),
        ):
            store.recover()

    def test_pointer_file_not_object(self) -> None:
        self.updater().refresh()
        with open(os.path.join(self.metadata_dir, "current"), "w") as f:
            f.write("[]")
        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ):
            generation = MetadataStore(self.metadata_dir).verify()
        # complete generation on disk is still adopted
        assert generation is not None

    def test_pointer_missing_generation_field(self) -> None:
        self.updater().refresh()
        with open(os.path.join(self.metadata_dir, "current"), "w") as f:
            json.dump({"format": 1}, f)
        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="ERROR"
        ):
            generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None

    def test_manifest_not_object(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        with open(os.path.join(generation.path, "manifest.json"), "w") as f:
            f.write("[]")
        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_manifest_generation_mismatch(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        manifest_path = os.path.join(generation.path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["generation"] = 99
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_manifest_files_missing(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        with open(os.path.join(generation.path, "manifest.json"), "w") as f:
            json.dump({"format": 1, "generation": generation.number}, f)
        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_manifest_entry_not_object(self) -> None:
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        manifest_path = os.path.join(generation.path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["files"]["targets"] = "nope"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        with self.assertRaises(exceptions.IncompleteMetadataGenerationError):
            MetadataStore(self.metadata_dir).verify()

    def test_stale_staging_and_temp_files_are_cleaned(self) -> None:
        self.updater().refresh()
        generations = os.path.join(self.metadata_dir, "generations")
        os.makedirs(os.path.join(generations, ".staging-dead"))
        stale_pointer = os.path.join(self.metadata_dir, ".current-dead.tmp")
        stale_anchor = os.path.join(self.metadata_dir, ".root-dead.tmp")
        for path in (stale_pointer, stale_anchor):
            with open(path, "w") as f:
                f.write("x")

        # recovery happens because a newer generation is on disk
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        self.publish()
        self.run_child("renamed")
        with self.assertLogs(
            "tuf.ngclient._internal.local_store", level="WARNING"
        ):
            MetadataStore(self.metadata_dir).verify()

        self.assertFalse(
            os.path.exists(os.path.join(generations, ".staging-dead"))
        )
        self.assertFalse(os.path.exists(stale_pointer))
        self.assertFalse(os.path.exists(stale_anchor))

    def test_failed_staging_is_cleaned_up(self) -> None:
        store = MetadataStore(self.metadata_dir)
        with (
            store.lock(),
            patch.object(os, "rename", side_effect=OSError("rename failed")),
            self.assertRaises(OSError),
        ):
            store.commit(None, {"targets": b"{}"}, {"targets": 1})
        generations = os.path.join(self.metadata_dir, "generations")
        # only a staging directory could exist, and it was removed
        self.assertEqual(
            [n for n in os.listdir(generations) if n.startswith(".staging")],
            [],
        )

    def test_owner_record_written_without_lock_is_noop(self) -> None:
        store = MetadataStore(self.metadata_dir)
        store._write_owner_record(heartbeat=True)  # no fd: no-op

    def test_holder_info_with_non_json_record(self) -> None:
        store = MetadataStore(
            self.metadata_dir, lock_timeout=0.0, lease_timeout=0.0
        )
        with open(store._lock_path, "wb") as f:
            f.write(b"not json")
        self.assertEqual(store._holder_info(), "lock holder is unknown")

    def test_holder_info_without_numeric_heartbeat(self) -> None:
        store = MetadataStore(self.metadata_dir, lock_timeout=0.0)
        with open(store._lock_path, "w") as f:
            json.dump({"pid": 1, "host": "h", "heartbeat": "old"}, f)
        info = store._holder_info()
        self.assertIn("held by pid 1", info)
        self.assertNotIn("heartbeat", info)

    def test_fsync_dir_noop_on_windows(self) -> None:
        with patch.object(local_store.os, "name", "nt"):
            local_store._fsync_dir(self.metadata_dir)

    def test_link_falls_back_to_copy(self) -> None:
        self.updater().refresh()
        source = os.path.join(
            utils.local_generation_path(self.metadata_dir), "targets.json"
        )
        destination = os.path.join(self.metadata_dir, "copied.json")
        with patch.object(os, "link", side_effect=OSError("no links")):
            local_store._link_or_copy(source, destination)
        self.assertTrue(os.path.exists(destination))

    def test_verify_digest_mismatch_and_no_fallback(self) -> None:
        """A manifest mismatch with no other generation is reported."""
        self.updater().refresh()
        store = MetadataStore(self.metadata_dir)
        visible = store.load()
        assert visible is not None
        mismatched = local_store.Generation(
            visible.number, visible.path, visible.roles, "0" * 64
        )
        original = store._load_generation

        def load_generation(
            number: int, verify_hashes: bool
        ) -> local_store.Generation:
            if verify_hashes and number == visible.number:
                return mismatched
            return original(number, verify_hashes)

        with (
            patch.object(
                store, "_load_generation", side_effect=load_generation
            ),
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            store.verify()

    def test_recover_raised_if_structural_load_fails_but_pointer_digest_ok(
        self,
    ) -> None:
        """A generation whose manifest cannot be parsed is discarded."""
        self.updater().refresh()
        generation = MetadataStore(self.metadata_dir).verify()
        assert generation is not None
        with open(os.path.join(generation.path, "manifest.json"), "w") as f:
            f.write("{")
        store = MetadataStore(self.metadata_dir)
        with (
            store.lock(),
            self.assertLogs(
                "tuf.ngclient._internal.local_store", level="ERROR"
            ),
            self.assertRaises(exceptions.IncompleteMetadataGenerationError),
        ):
            store.recover()


if __name__ == "__main__":
    utils.configure_test_logging(sys.argv)
    unittest.main()
