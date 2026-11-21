# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for the append-only, tamper-evident update audit log.

The audit log is a pure observability side-channel of the update workflow:
these tests verify that it records a complete evidence chain (metadata
versions and hashes, signing key identifiers, rejection reasons, target
hash binding) while the updater's existing behavior stays unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from tests.repository_simulator import RepositorySimulator
from tuf.api.exceptions import (
    BadVersionNumberError,
    ExpiredMetadataError,
    LengthOrHashMismatchError,
    UnsignedMetadataError,
)
from tuf.api.metadata import DelegatedRole, Targets
from tuf.ngclient import (
    AuditLog,
    AuditLogFormatError,
    AuditLogVersionError,
    Updater,
    UpdaterConfig,
)
from tuf.ngclient.audit import (
    AUDIT_SCHEMA_VERSION,
    RULE_HASH_MISMATCH,
    RULE_ROLLBACK,
    RULE_SIGNATURES,
    verify_target_file,
)

if TYPE_CHECKING:
    from tuf.ngclient.audit import UpdateAuditor


class AuditLogTestBase(unittest.TestCase):
    """Wire a RepositorySimulator to an Updater with an audit log."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        os.mkdir(self.metadata_dir)
        os.mkdir(self.targets_dir)
        self.audit_path = os.path.join(self.temp_dir.name, "audit.log")
        self.sim = RepositorySimulator()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _updater(self, *, audit: bool = True) -> Updater:
        config = (
            UpdaterConfig(audit_log_path=self.audit_path)
            if audit
            else UpdaterConfig()
        )
        return Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            self.sim,
            config=config,
            bootstrap=self.sim.signed_roots[0],
        )

    def _records(self) -> list[dict]:
        return AuditLog.read_all(self.audit_path)

    def _refresh(self, updater: Updater) -> None:
        updater.refresh()
        updater.close_audit_log()

    def _role_events(self, records: list[dict], role: str) -> list[dict]:
        return [
            event
            for record in records
            for event in record["role_events"]
            if event["role"] == role
        ]


class TestAuditEvidenceChain(AuditLogTestBase):
    """Required evidence-chain scenarios."""

    def test_successful_update_records_full_evidence(self) -> None:
        # A delegated role signs the target so the record must cover root,
        # timestamp, snapshot, top-level targets and the delegate.
        role = DelegatedRole("delegated", [], 1, False, ["*"])
        self.sim.add_delegation(
            "targets", role, Targets(expires=self.sim.safe_expiry)
        )
        self.sim.add_target("delegated", b"firmware payload", "fw.bin")
        self.sim.update_snapshot()

        updater = self._updater()
        info = updater.get_targetinfo("fw.bin")
        self.assertIsNotNone(info)
        assert info is not None
        path = updater.download_target(info)
        updater.close_audit_log()
        self.assertEqual(updater.audit_write_failures, [])

        records = self._records()
        self.assertTrue(records)

        # Every role used in the decision has a version, content hash and
        # signing key identifiers.
        seen_roles = {
            event["role"]
            for record in records
            for event in record["role_events"]
        }
        self.assertEqual(
            seen_roles,
            {"root", "timestamp", "snapshot", "targets", "delegated"},
        )
        for event in (
            event for record in records for event in record["role_events"]
        ):
            self.assertEqual(event["outcome"], "accepted")
            self.assertIn("version", event)
            self.assertRegex(event["content_hash"]["sha256"], r"^[0-9a-f]{64}$")
            sig = event["signatures"]
            # root evaluations are a list (previous + new root); the rest
            # are single threshold evaluations.
            evaluations = sig if isinstance(sig, list) else [sig]
            for evaluation in evaluations:
                self.assertGreaterEqual(evaluation["threshold"], 1)
                self.assertEqual(
                    evaluation["received_signatures"],
                    len(evaluation["signed_keyids"]),
                )
                self.assertTrue(evaluation["threshold_met"])
                for key in evaluation["keys"]:
                    # Only identifiers and fingerprints, never key material.
                    self.assertNotIn("keyval", key)
                    self.assertRegex(key["fingerprint"], r"^[0-9a-f]{64}$")

        # Records are append-only and chained in sequence.
        self.assertEqual(
            [record["seq"] for record in records],
            list(range(1, len(records) + 1)),
        )

        # The download record binds metadata hashes to the on-disk file.
        download_records = [r for r in records if "download" in r]
        self.assertEqual(len(download_records), 1)
        download = download_records[0]["download"]
        self.assertEqual(download["target"], "fw.bin")
        self.assertTrue(download["hash_verified"])
        self.assertTrue(verify_target_file(download_records[0], path))

    def test_insufficient_signatures_rejection(self) -> None:
        # Timestamp metadata with no signatures fails threshold verification.
        self.sim.signers["timestamp"].clear()
        updater = self._updater()
        with self.assertRaises(UnsignedMetadataError):
            updater.refresh()
        updater.close_audit_log()

        records = self._records()
        rejected = self._role_events(records, "timestamp")
        self.assertTrue(
            any(event["outcome"] == "rejected" for event in rejected)
        )
        event = next(
            event for event in rejected if event["outcome"] == "rejected"
        )
        self.assertEqual(event["rejection_rule"], RULE_SIGNATURES)
        self.assertEqual(event["signatures"]["threshold"], 1)
        self.assertEqual(event["signatures"]["received_signatures"], 0)
        self.assertEqual(event["signatures"]["signed_keyids"], [])
        # The session record names the failed stage and original error type.
        final = records[-1]
        self.assertEqual(final["outcome"], "rejected")
        self.assertEqual(final["error_type"], "UnsignedMetadataError")

    def test_version_rollback_rejection(self) -> None:
        # Prime a trusted timestamp v2, then serve v1.
        self.sim.timestamp.version = 2
        updater = self._updater()
        updater.refresh()
        updater.close_audit_log()

        self.sim.timestamp.version = 1
        updater2 = self._updater()
        with self.assertRaises(BadVersionNumberError):
            updater2.refresh()
        updater2.close_audit_log()

        records = self._records()
        rollback_events = [
            event
            for event in self._role_events(records, "timestamp")
            if event.get("rejection_rule") == RULE_ROLLBACK
        ]
        self.assertEqual(len(rollback_events), 1)
        self.assertEqual(rollback_events[0]["outcome"], "rejected")
        # Failure record pinpoints role and rule.
        final = records[-1]
        self.assertEqual(final["rejection_rule"], RULE_ROLLBACK)
        self.assertEqual(final["failed_stage"], "refresh")

    def test_hash_mismatch_rejection(self) -> None:
        # Pin the snapshot hash in timestamp, then change snapshot bytes
        # without updating the pinned hash.
        self.sim.compute_metafile_hashes_length = True
        self.sim.update_timestamp()
        updater = self._updater()
        updater.refresh()
        updater.close_audit_log()

        self.sim.snapshot.expires += timedelta(days=1)
        self.sim.snapshot.version += 1
        self.sim.timestamp.snapshot_meta.version = self.sim.snapshot.version
        self.sim.timestamp.version += 1

        updater2 = self._updater()
        with self.assertRaises(LengthOrHashMismatchError):
            updater2.refresh()
        updater2.close_audit_log()

        records = self._records()
        mismatch = [
            event
            for event in self._role_events(records, "snapshot")
            if event.get("rejection_rule") == RULE_HASH_MISMATCH
        ]
        self.assertEqual(len(mismatch), 1)
        self.assertIn("content_hash", mismatch[0])
        self.assertEqual(records[-1]["rejection_rule"], RULE_HASH_MISMATCH)


class TestAuditHashBinding(AuditLogTestBase):
    def test_record_hash_matches_downloaded_file(self) -> None:
        payload = b"binding-test-firmware"
        self.sim.add_target("targets", payload, "bound.bin")
        self.sim.update_snapshot()

        updater = self._updater()
        info = updater.get_targetinfo("bound.bin")
        self.assertIsNotNone(info)
        assert info is not None
        path = updater.download_target(info)
        updater.close_audit_log()

        records = self._records()
        record = next(r for r in records if "download" in r)
        # Independently hash the actual file and compare against the record.
        with open(path, "rb") as target_file:
            digest = hashlib.sha256(target_file.read()).hexdigest()
        self.assertEqual(
            digest, record["download"]["metadata_hashes"]["sha256"]
        )
        self.assertEqual(
            digest, record["download"]["downloaded_hashes"]["sha256"]
        )
        self.assertTrue(verify_target_file(record, path))


class TestAuditTamperEvidence(AuditLogTestBase):
    def test_tampered_record_is_detected_and_located(self) -> None:
        self.sim.add_target("targets", b"x", "a.bin")
        self.sim.update_snapshot()
        updater = self._updater()
        info = updater.get_targetinfo("a.bin")
        self.assertIsNotNone(info)
        assert info is not None
        updater.download_target(info)
        updater.close_audit_log()

        # Hand-modify a field inside the second record without recomputing
        # the record hash.
        with open(self.audit_path) as log_file:
            lines = log_file.read().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        target_line = json.loads(lines[1])
        target_line["detail"] = "forged detail"
        lines[1] = json.dumps(
            target_line, sort_keys=True, separators=(",", ":")
        )
        with open(self.audit_path, "w") as log_file:
            log_file.write("\n".join(lines) + "\n")

        with self.assertRaises(AuditLogFormatError) as context:
            AuditLog.read_all(self.audit_path)
        # The offending record is pinpointed by sequence and line number.
        self.assertEqual(context.exception.seq, target_line["seq"])
        self.assertEqual(context.exception.line_number, 2)

        # Lenient read still returns every intact record and reports the
        # corruption instead of silently dropping it.
        state = AuditLog.read_lenient(self.audit_path)
        self.assertEqual(len(state.corruptions), 1)
        self.assertEqual(state.corruptions[0].seq, target_line["seq"])
        self.assertTrue(state.records)

    def test_truncated_log_is_reported_with_line(self) -> None:
        self.sim.add_target("targets", b"x", "a.bin")
        self.sim.update_snapshot()
        updater = self._updater()
        info = updater.get_targetinfo("a.bin")
        self.assertIsNotNone(info)
        assert info is not None
        updater.download_target(info)
        updater.close_audit_log()

        with open(self.audit_path, "rb") as log_file:
            content = log_file.read()
        # Chop the final line in half.
        with open(self.audit_path, "wb") as log_file:
            log_file.write(content[: len(content) - 5])

        state = AuditLog.read_lenient(self.audit_path)
        self.assertEqual(len(state.corruptions), 1)
        self.assertIn("not valid JSON", state.corruptions[0].reason)


class TestAuditFailureTrace(AuditLogTestBase):
    def test_failed_update_records_where_it_stopped(self) -> None:
        # Expired timestamp: refresh must fail and the record must show the
        # stopping stage and the metadata state at rejection time.
        self.sim.timestamp.expires = datetime.now(UTC) - timedelta(days=1)
        self.sim.update_timestamp()

        updater = self._updater()
        with self.assertRaises(ExpiredMetadataError):
            updater.refresh()
        updater.close_audit_log()

        records = self._records()
        final = records[-1]
        self.assertEqual(final["outcome"], "rejected")
        self.assertEqual(final["failed_stage"], "refresh")
        self.assertEqual(final["rejection_rule"], "metadata_expired")
        self.assertEqual(final["error_type"], "ExpiredMetadataError")
        # The rejected timestamp event still carries the observed bytes hash.
        timestamp_events = self._role_events(records, "timestamp")
        self.assertTrue(
            all("content_hash" in event for event in timestamp_events)
        )


class TestAuditOperationalBehavior(AuditLogTestBase):
    def test_audit_disabled_by_default_changes_nothing(self) -> None:
        # No audit_log_path: no log file is created and the workflow runs
        # exactly as before.
        updater = self._updater(audit=False)
        updater.refresh()
        info = updater.get_targetinfo("some-missing-target")
        self.assertIsNone(info)
        self.assertIsNone(updater.audit_log_path)
        self.assertEqual(updater.audit_write_failures, [])
        self.assertFalse(os.path.exists(self.audit_path))

    def test_query_by_target_and_time(self) -> None:
        self.sim.add_target("targets", b"one", "one.bin")
        self.sim.add_target("targets", b"two", "two.bin")
        self.sim.update_snapshot()
        updater = self._updater()
        for name in ("one.bin", "two.bin"):
            info = updater.get_targetinfo(name)
            self.assertIsNotNone(info)
            assert info is not None
            updater.download_target(info)
        updater.close_audit_log()

        one = AuditLog.query(self.audit_path, target="one.bin")
        self.assertTrue(one)
        self.assertTrue(all(record["target"] == "one.bin" for record in one))
        future = datetime.now(UTC) + timedelta(days=1)
        self.assertEqual(AuditLog.query(self.audit_path, start=future), [])

    def test_incompatible_schema_version_is_rejected(self) -> None:
        self.sim.add_target("targets", b"x", "a.bin")
        self.sim.update_snapshot()
        updater = self._updater()
        info = updater.get_targetinfo("a.bin")
        self.assertIsNotNone(info)
        assert info is not None
        updater.download_target(info)
        updater.close_audit_log()

        with open(self.audit_path) as log_file:
            lines = log_file.read().splitlines()
        record = json.loads(lines[0])
        record["schema_version"] = AUDIT_SCHEMA_VERSION + 99
        lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with open(self.audit_path, "w") as log_file:
            log_file.write("\n".join(lines) + "\n")

        with self.assertRaises(AuditLogVersionError) as context:
            AuditLog.read_all(self.audit_path)
        self.assertEqual(
            context.exception.found_version,
            AUDIT_SCHEMA_VERSION + 99,
        )

    def test_write_failure_is_surfaced_not_silent(self) -> None:
        updater = self._updater()
        # Close the log so further writes fail; the update must still
        # succeed and report the missing audit record.
        auditor: UpdateAuditor = updater._auditor  # type: ignore[assignment]
        auditor.close()

        updater.refresh()
        # A new session cannot be persisted after close.
        self.assertTrue(updater.audit_write_failures)

    def test_append_keeps_existing_chain(self) -> None:
        # First updater writes records; a second updater appends to the same
        # log and the full chain still verifies.
        self.sim.add_target("targets", b"first", "first.bin")
        self.sim.update_snapshot()
        first = self._updater()
        info = first.get_targetinfo("first.bin")
        self.assertIsNotNone(info)
        assert info is not None
        first.download_target(info)
        first.close_audit_log()
        first_count = len(self._records())

        second = self._updater()
        second.refresh()
        second.close_audit_log()

        records = self._records()
        self.assertGreater(len(records), first_count)
        self.assertEqual(
            [record["seq"] for record in records],
            list(range(1, len(records) + 1)),
        )


if __name__ == "__main__":
    unittest.main()
