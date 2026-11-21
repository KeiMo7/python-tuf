# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Append-only, tamper-evident audit log for the client update workflow.

The audit log is an *observability side-channel*: it records the evidence
chain of an update (which metadata versions and content hashes were used,
which keys signed each role, why a rejection happened and which target file
was downloaded) without ever participating in the update decision. The
update workflow's validation order, rejection conditions, return values and
exception types are unchanged.

The log is enabled with ``UpdaterConfig(audit_log_path=...)``: when no path
is configured, no audit code runs and no overhead is added.

On-disk format
--------------
The log is a UTF-8 text file with one JSON object per line (JSON Lines).
Every line is self describing and independently parseable. Each record
contains:

* ``schema_version``: log format version (currently 1)
* ``seq``: monotonically increasing record number, starting at 1
* ``timestamp``: UTC ISO-8601 timestamp of the event
* ``prev_hash``: SHA-256 hex digest of the previous record line
* ``record_hash``: SHA-256 hex digest of the record line with
  ``record_hash`` replaced by an empty string

A record whose ``record_hash`` no longer matches, or whose ``prev_hash``
does not match the previous record's ``record_hash``, has been tampered
with: verification reports the sequence number of the offending record and
keeps all other records readable.

Records only contain key identifiers (keyids and non-reversible key
fingerprints), never key material.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from tuf.api import exceptions
from tuf.api.metadata import Metadata, Root, Targets

if TYPE_CHECKING:
    from securesystemslib.signer import Key, Signature

    from tuf.ngclient._internal.trusted_metadata_set import (
        AcceptedMetadata,
        RejectedMetadata,
    )

logger = logging.getLogger(__name__)

# Structural type of the trusted set: the auditor only needs role lookup.
if TYPE_CHECKING:

    class _RoleLookup(Protocol):
        def get(self, role: str) -> Root | Targets | None: ...


AUDIT_SCHEMA_VERSION = 1

# Outcome markers
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"

# Rejection rule identifiers: they map 1:1 to the rejection conditions the
# update workflow actually raises on. The strings are identical to the rule
# constants in trusted_metadata_set.
RULE_ROLLBACK = "version_rollback"
RULE_EXPIRED = "metadata_expired"
RULE_HASH_MISMATCH = "hash_mismatch"
RULE_SIGNATURES = "insufficient_signatures"
RULE_ROLE_MISSING = "role_missing"
RULE_MALFORMED = "malformed_metadata"
RULE_DOWNLOAD = "download_failed"
RULE_TARGET_HASH = "target_hash_mismatch"
RULE_TARGET_WRITE = "target_write_failed"


def classify_rejection(exc: Exception) -> str:
    """Map an exception raised by the update workflow to a rule identifier.

    The mapping is derived from the existing TUF exception types so that the
    audit layer never has to interpret exception messages.
    """
    rule = _EXCEPTION_RULES.get(type(exc))
    if rule is not None:
        return rule
    # RepositoryError subclasses not listed explicitly are malformed
    # metadata; DownloadError subclasses are download failures.
    if isinstance(exc, exceptions.RepositoryError):
        return RULE_MALFORMED
    if isinstance(exc, exceptions.DownloadError):
        return RULE_DOWNLOAD
    return type(exc).__name__


_EXCEPTION_RULES: dict[type[Exception], str] = {
    exceptions.UnsignedMetadataError: RULE_SIGNATURES,
    exceptions.ExpiredMetadataError: RULE_EXPIRED,
    exceptions.BadVersionNumberError: RULE_ROLLBACK,
    exceptions.EqualVersionNumberError: RULE_ROLLBACK,
    exceptions.LengthOrHashMismatchError: RULE_HASH_MISMATCH,
}


def key_fingerprint(key: Key) -> str:
    """Return a non-reversible fingerprint of a public key.

    Only the fingerprint and the keyid are ever written to the log: the key
    material (``key.keyval``) is never serialized.
    """
    material = json.dumps(
        {
            "keyid": key.keyid,
            "keytype": key.keytype,
            "scheme": key.scheme,
            "keyval": key.keyval,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def evaluate_signatures(
    delegator: Root | Targets,
    role_name: str,
    payload: bytes,
    signatures: dict[str, Signature],
) -> dict[str, Any]:
    """Describe the signature situation of one role's metadata.

    The result contains the delegated role's threshold, every keyid the
    delegator trusts for this role (marked signed/unsigned/invalid), every
    untrusted keyid that presented a signature, and a boolean threshold
    verdict. No private or public key material is included.
    """

    role = delegator.get_delegated_role(role_name)

    signed: list[str] = []
    unsigned: list[str] = []
    invalid: list[str] = []
    keys: list[dict[str, str]] = []

    for keyid in role.keyids:
        try:
            key = delegator.get_key(keyid)
        except ValueError:
            # keyid is listed for the role but delegator has no such key:
            # it cannot produce a signature the role accepts.
            unsigned.append(keyid)
            continue

        keys.append(
            {
                "keyid": key.keyid,
                "keytype": key.keytype,
                "scheme": key.scheme,
                "fingerprint": key_fingerprint(key),
            }
        )

        sig = signatures.get(keyid)
        if sig is None:
            unsigned.append(keyid)
            continue

        try:
            key.verify_signature(sig, payload)
        except Exception:  # noqa: BLE001
            invalid.append(keyid)
        else:
            signed.append(keyid)

    untrusted = sorted(
        keyid for keyid in signatures if keyid not in role.keyids
    )

    return {
        "threshold": role.threshold,
        "required_keyids": len(role.keyids),
        "received_signatures": len(signatures),
        "signed_keyids": signed,
        "unsigned_keyids": unsigned,
        "invalid_keyids": invalid,
        "untrusted_keyids": untrusted,
        "keys": keys,
        "threshold_met": len(signed) >= role.threshold,
    }


def evaluate_root_signatures(
    previous: Root | None,
    new_root: Root,
    payload: bytes,
    signatures: dict[str, Signature],
) -> list[dict[str, Any]]:
    """Describe root signature verification against both root versions.

    A new root version must be signed by the previous root and (for the
    self-consistent trust anchor) by itself. Both evaluations are recorded.
    """
    results = []
    if previous is not None and previous.version != new_root.version:
        results.append(
            evaluate_signatures(previous, Root.type, payload, signatures)
            | {"verified_against": f"root v{previous.version}"}
        )
    results.append(
        evaluate_signatures(new_root, Root.type, payload, signatures)
        | {"verified_against": f"root v{new_root.version}"}
    )
    return results


@dataclass
class AuditCorruption:
    """Describes one unreadable or tampered log line."""

    seq: int
    line_number: int
    reason: str


@dataclass
class AuditLogState:
    """Result of reading a log that may contain corrupt records."""

    records: list[dict[str, Any]] = field(default_factory=list)
    corruptions: list[AuditCorruption] = field(default_factory=list)


@dataclass
class _LineVerification:
    """Outcome of verifying a single audit log line."""

    record: dict[str, Any] | None = None
    corruption: AuditCorruption | None = None
    incompatible_schema: AuditLogVersionError | None = None


class AuditLogError(Exception):
    """Base class for audit log storage and format errors."""


class AuditLogWriteError(AuditLogError):
    """An audit record could not be persisted.

    The update workflow itself is unaffected: this error is surfaced through
    ``Updater.audit_write_failures`` rather than raised into the workflow.
    """


class AuditLogFormatError(AuditLogError):
    """The log file is malformed, truncated or otherwise unreadable."""

    def __init__(self, message: str, *, line_number: int, seq: int | None):
        super().__init__(message)
        self.line_number = line_number
        self.seq = seq


class AuditLogVersionError(AuditLogError):
    """The log was written by an incompatible audit log implementation."""

    def __init__(self, found: int, supported: int):
        super().__init__(
            f"Incompatible audit log schema_version {found}: "
            f"this implementation supports version {supported}"
        )
        self.found_version = found
        self.supported_version = supported


class AuditLog:
    """Append-only, tamper-evident audit log storage.

    Writes are serialized through a background thread and an in-memory
    queue so that log IO never blocks the update workflow: hashing and
    JSON serialization happen on the calling thread (cheap CPU work), only
    file IO is offloaded. If a write fails, the error is captured and
    reported through :attr:`write_failures` instead of being raised.

    Each record is one JSON line; records are chained with SHA-256 hashes
    so that external modification or truncation can be detected and located
    down to a specific record.
    """

    def __init__(self, path: str):
        self._path = path
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._seq = 0
        self._prev_hash = ""
        self._closed = False
        self.write_failures: list[str] = []

        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        # Ensure the file exists even if no record is ever persisted (e.g.
        # when the very first update is rejected): readers then see an
        # empty log rather than a missing file.
        if not os.path.exists(path):
            open(path, "ab").close()
        if os.path.getsize(path) > 0:
            self._restore_chain_state(path)

        self._thread = threading.Thread(
            target=self._writer_loop,
            name="tuf-audit-log",
            daemon=True,
        )
        self._thread.start()

    def _restore_chain_state(self, path: str) -> None:
        """Recover seq and prev_hash from an existing, internally valid log."""
        last_hash = ""
        last_seq = 0
        with open(path, "rb") as log_file:
            for line_number, line in enumerate(log_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    seq = int(record["seq"])
                    record_hash = str(record["record_hash"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise AuditLogFormatError(
                        f"Cannot continue existing audit log: line "
                        f"{line_number} is not a valid record: {exc}",
                        line_number=line_number,
                        seq=None,
                    ) from exc
                if record.get("prev_hash") != last_hash:
                    raise AuditLogFormatError(
                        f"Cannot continue existing audit log: hash chain "
                        f"broken at line {line_number} (seq {seq})",
                        line_number=line_number,
                        seq=seq,
                    )
                last_hash = record_hash
                last_seq = seq
        self._seq = last_seq
        self._prev_hash = last_hash

    def _writer_loop(self) -> None:
        """Background writer: append serialized records, capture IO errors."""
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with open(self._path, "ab") as log_file:
                    log_file.write(item)
                    log_file.flush()
                    os.fsync(log_file.fileno())
            except OSError as exc:
                self.write_failures.append(str(exc))
            finally:
                self._queue.task_done()

    def append(self, record: dict[str, Any]) -> None:
        """Hash-chain, serialize and enqueue a record for background write.

        Raises:
            AuditLogWriteError: the log has been closed.
        """
        with self._lock:
            if self._closed:
                # A record that cannot be persisted is a write failure:
                # surface it like a background IO failure, then refuse.
                message = "audit log is closed: record was not persisted"
                if message not in self.write_failures:
                    self.write_failures.append(message)
                raise AuditLogWriteError(message)
            self._seq += 1
            entry = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "seq": self._seq,
                "timestamp": datetime.now(UTC)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                "prev_hash": self._prev_hash,
                **record,
            }
            # The record hash covers the full record with record_hash
            # present but empty: this is the exact form _rehash() uses when
            # verifying an existing record.
            entry["record_hash"] = ""
            line = json.dumps(
                entry, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            record_hash = hashlib.sha256(line).hexdigest()
            entry["record_hash"] = record_hash
            line = json.dumps(
                entry, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            self._prev_hash = record_hash
        self._queue.put(line + b"\n")

    def close(self) -> None:
        """Wait for queued records to be flushed and stop the writer thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()

    @property
    def path(self) -> str:
        """Filesystem path of the audit log."""
        return self._path

    @staticmethod
    def _rehash(record: dict[str, Any]) -> str:
        entry = dict(record)
        entry["record_hash"] = ""
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(line).hexdigest()

    @classmethod
    def read_all(cls, path: str) -> list[dict[str, Any]]:
        """Read and fully verify a log, raising on the first problem.

        Raises:
            AuditLogFormatError: a line is malformed/truncated or the hash
                chain is broken. The error identifies the offending line
                and record sequence number.
            AuditLogVersionError: the log uses an incompatible schema
                version.
        """
        state = cls.read_lenient(path)
        if state.corruptions:
            corruption = state.corruptions[0]
            raise AuditLogFormatError(
                corruption.reason,
                line_number=corruption.line_number,
                seq=corruption.seq,
            )
        return state.records

    @classmethod
    def read_lenient(cls, path: str) -> AuditLogState:
        """Read a log tolerating corrupt records.

        Every parseable record with an intact hash chain is returned; lines
        that are malformed, truncated or break the chain are reported in
        ``corruptions`` (with line number and sequence number when known)
        instead of aborting the read.
        """
        state = AuditLogState()
        expected_prev = ""
        version_checked = False

        try:
            log_file = open(path, "rb")  # noqa: SIM115
        except OSError as exc:
            raise AuditLogFormatError(
                f"audit log cannot be opened: {exc}",
                line_number=0,
                seq=None,
            ) from exc

        with log_file:
            for line_number, raw_line in enumerate(log_file, start=1):
                if not raw_line.strip():
                    continue
                outcome = cls._verify_line(
                    raw_line, line_number, expected_prev, version_checked
                )
                if outcome.incompatible_schema is not None:
                    raise outcome.incompatible_schema
                if outcome.corruption is not None:
                    state.corruptions.append(outcome.corruption)
                    # The chain cannot be continued after a broken line.
                    break
                # No corruption or schema error means a record is present.
                if outcome.record is None:  # pragma: no cover
                    msg = "verification returned neither record nor error"
                    raise RuntimeError(msg)
                version_checked = True
                state.records.append(outcome.record)
                expected_prev = outcome.record["record_hash"]

        return state

    @classmethod
    def _verify_line(
        cls,
        raw_line: bytes,
        line_number: int,
        expected_prev: str,
        version_checked: bool,
    ) -> _LineVerification:
        """Verify one log line against the running hash chain."""

        def corrupted(seq: int, reason: str) -> _LineVerification:
            return _LineVerification(
                corruption=AuditCorruption(seq, line_number, reason)
            )

        try:
            record = json.loads(raw_line)
        except ValueError as exc:
            return corrupted(
                -1,
                f"line {line_number} is not valid JSON "
                f"(truncated or corrupted): {exc}",
            )

        seq = record.get("seq")
        if not isinstance(seq, int):
            return corrupted(-1, f"line {line_number} has no integer 'seq'")

        schema = record.get("schema_version")
        if not version_checked and (
            not isinstance(schema, int) or schema > AUDIT_SCHEMA_VERSION
        ):
            found = schema if isinstance(schema, int) else -1
            return _LineVerification(
                incompatible_schema=AuditLogVersionError(
                    found, AUDIT_SCHEMA_VERSION
                )
            )

        return cls._verify_record_hash_chain(
            record, seq, line_number, expected_prev
        )

    @classmethod
    def _verify_record_hash_chain(
        cls,
        record: dict[str, Any],
        seq: int,
        line_number: int,
        expected_prev: str,
    ) -> _LineVerification:
        """Check record_hash presence, chain linkage and content hash."""

        def corrupted(reason: str) -> _LineVerification:
            return _LineVerification(
                corruption=AuditCorruption(seq, line_number, reason)
            )

        stored_hash = record.get("record_hash")
        if not isinstance(stored_hash, str):
            return corrupted(
                f"record seq {seq} (line {line_number}) has no record_hash"
            )

        if record.get("prev_hash") != expected_prev:
            return corrupted(
                f"hash chain broken at record seq {seq} (line "
                f"{line_number}): prev_hash does not match the previous "
                "record"
            )

        if cls._rehash(record) != stored_hash:
            return corrupted(
                f"record seq {seq} (line {line_number}) was modified: "
                "record_hash verification failed"
            )

        return _LineVerification(record=record)

    @classmethod
    def query(
        cls,
        path: str,
        *,
        target: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return verified records, optionally filtered by target and time.

        ``start`` and ``end`` are compared against record timestamps (both
        bounds optional, UTC aware datetimes expected).
        """
        records = cls.read_all(path)
        result = []
        for record in records:
            if target is not None and record.get("target") != target:
                continue
            if start is not None or end is not None:
                timestamp_text = str(record["timestamp"])
                if timestamp_text.endswith("Z"):
                    timestamp = datetime.fromisoformat(
                        timestamp_text.removesuffix("Z")
                    ).replace(tzinfo=UTC)
                else:
                    timestamp = datetime.fromisoformat(timestamp_text)
                if start is not None and timestamp < start:
                    continue
                if end is not None and timestamp > end:
                    continue
            result.append(record)
        return result


class UpdateAuditor:
    """Observes one Updater instance and turns events into audit records.

    The auditor owns the :class:`AuditLog` and the state of the update
    session currently in progress (metadata role events accumulate until
    the session is finalized on download success, workflow failure or an
    explicit ``refresh()`` completion). All public methods swallow their
    own errors so that auditing can never alter the update workflow.
    """

    def __init__(self, path: str):
        self.log = AuditLog(path)
        self._session: dict[str, Any] | None = None
        self._session_seq = 0
        self._trusted_set: _RoleLookup | None = None

    @property
    def write_failures(self) -> list[str]:
        """Storage errors collected while writing audit records."""
        return self.log.write_failures

    def close(self) -> None:
        """Flush and close the underlying audit log."""
        self.log.close()

    def begin(self, operation: str, target: str | None = None) -> None:
        """Start a new update session, finalizing any open one first."""
        if self._session is not None:
            self.finalize(
                OUTCOME_ACCEPTED,
                "session closed without explicit finalization",
            )
        self._session_seq += 1
        self._session = {
            "session": self._session_seq,
            "operation": operation,
            "target": target,
            "events": [],
        }

    @property
    def has_open_session(self) -> bool:
        """Whether an update session is currently accumulating events."""
        return self._session is not None

    def bind_trusted_set(self, trusted_set: _RoleLookup) -> None:
        """Register the trusted set used to look up delegator payloads.

        Needed to describe the signature situation of a rejected role
        without re-running verification.
        """
        self._trusted_set = trusted_set

    def _ensure_session(self, operation: str) -> dict[str, Any]:
        if self._session is None:
            self.begin(operation)
        if self._session is None:  # pragma: no cover
            msg = "session creation did not open a session"
            raise RuntimeError(msg)
        return self._session

    def _record_event(self, entry: dict[str, Any]) -> None:
        try:
            session = self._ensure_session("metadata_load")
            session["events"].append(entry)
        except Exception:  # noqa: BLE001
            # Auditing must never interfere with the update workflow.
            logger.debug("audit event recording failed", exc_info=True)

    def record_download(
        self,
        target: str,
        targetinfo_hashes: dict[str, str],
        downloaded_hashes: dict[str, str],
        length: int,
        filepath: str,
    ) -> None:
        """Record the target download and its hash binding to disk."""
        try:
            session = self._ensure_session("download_target")
            session["target"] = target
            session["download"] = {
                "target": target,
                "filepath": filepath,
                "length": length,
                "metadata_hashes": targetinfo_hashes,
                "downloaded_hashes": downloaded_hashes,
                "hash_verified": targetinfo_hashes == downloaded_hashes,
            }
        except Exception:  # noqa: BLE001
            logger.debug("audit operation failed", exc_info=True)

    def finalize(
        self,
        outcome: str,
        detail: str | None = None,
        *,
        rule: str | None = None,
        error_type: str | None = None,
        stage: str | None = None,
    ) -> None:
        """Write the open session as one audit record and close it."""
        session = self._session
        if session is None:
            return
        self._session = None
        try:
            record: dict[str, Any] = {
                "session": session["session"],
                "operation": session["operation"],
                "target": session["target"],
                "outcome": outcome,
                "role_events": session["events"],
            }
            if "download" in session:
                record["download"] = session["download"]
            if detail is not None:
                record["detail"] = detail
            if rule is not None:
                record["rejection_rule"] = rule
            if error_type is not None:
                record["error_type"] = error_type
            if stage is not None:
                record["failed_stage"] = stage
            self.log.append(record)
        except Exception:  # noqa: BLE001
            logger.debug("audit operation failed", exc_info=True)

    # AuditHook implementation used by TrustedMetadataSet
    def metadata_accepted(self, event: AcceptedMetadata) -> None:
        """Handle a successful metadata verification event."""
        try:
            if event.role == Root.type:
                signatures_info: Any = evaluate_root_signatures(
                    event.previous_root,
                    cast("Root", event.payload),
                    event.payload_bytes,
                    event.signatures,
                )
            elif event.delegator is not None:
                signatures_info = evaluate_signatures(
                    event.delegator,
                    event.role,
                    event.payload_bytes,
                    event.signatures,
                )
            else:
                signatures_info = None

            self._record_event(
                {
                    "event": "metadata_load",
                    "role": event.role,
                    "delegator": event.delegator_name,
                    "source": event.source,
                    "outcome": OUTCOME_ACCEPTED,
                    "version": event.payload.version,
                    "content_hash": {
                        "sha256": hashlib.sha256(event.raw_data).hexdigest()
                    },
                    "signatures": signatures_info,
                }
            )
        except Exception:  # noqa: BLE001
            logger.debug("audit operation failed", exc_info=True)

    def metadata_rejected(self, event: RejectedMetadata) -> None:
        """Handle a failed metadata verification event."""
        try:
            entry: dict[str, Any] = {
                "event": "metadata_load",
                "role": event.role,
                "delegator": event.delegator_name,
                "source": event.source,
                "outcome": OUTCOME_REJECTED,
                "rejection_rule": event.rule,
                "error": str(event.error),
            }
            if event.raw_data is not None:
                entry["content_hash"] = {
                    "sha256": hashlib.sha256(event.raw_data).hexdigest()
                }
                # Best-effort signature detail for threshold failures:
                # get_verification_result style evaluation never raises on
                # an unmet threshold.
                if event.rule == RULE_SIGNATURES:
                    detail = self._rejected_signature_detail(event)
                    if detail is not None:
                        entry["signatures"] = detail[0]
                        entry["version"] = detail[1]
            self._record_event(entry)
        except Exception:  # noqa: BLE001
            logger.debug("audit operation failed", exc_info=True)

    def _rejected_signature_detail(
        self, event: RejectedMetadata
    ) -> tuple[dict[str, Any], int | None] | None:
        """Re-evaluate signatures of a rejected role without raising."""
        try:
            md = Metadata.from_bytes(event.raw_data) if event.raw_data else None
            if md is None:
                return None
            delegator = self._delegator_for(event.delegator_name)
            if delegator is None:
                return None
            info = evaluate_signatures(
                delegator,
                event.role,
                md.signed_bytes,
                md.signatures,
            )
            return info, md.signed.version
        except Exception:  # noqa: BLE001
            return None

    def _delegator_for(self, delegator_name: str) -> Root | Targets | None:
        """Return the delegator payload from the bound trusted set."""
        if self._trusted_set is None:
            return None
        return self._trusted_set.get(delegator_name)


def verify_target_file(record: dict[str, Any], filepath: str) -> bool:
    """Re-verify a downloaded file against the hashes stored in an audit
    record. Returns True only if every recorded hash matches the file.
    """
    download = record.get("download")
    if not download:
        raise ValueError("record contains no download evidence")
    expected = download["metadata_hashes"]
    with open(filepath, "rb") as target_file:
        content = target_file.read()
    actual = {
        algorithm: hashlib.new(algorithm, content).hexdigest()
        for algorithm in expected
    }
    return actual == expected
