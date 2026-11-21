# Copyright 2021-2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Trusted collection of client-side TUF Metadata.

``TrustedMetadataSet`` keeps track of the current valid set of metadata for the
client, and handles almost every step of the "Detailed client workflow" (
https://theupdateframework.github.io/specification/latest#detailed-client-workflow)
in the TUF specification: the remaining steps are related to filesystem and
network IO, which are not handled here.

Loaded metadata can be accessed via index access with rolename as key
(``trusted_set[Root.type]``) or, in the case of top-level metadata, using the
helper properties (``trusted_set.root``).

Signatures are verified and discarded upon inclusion into the trusted set.

The rules that ``TrustedMetadataSet`` follows for top-level metadata are
 * Metadata must be loaded in order:
   root -> timestamp -> snapshot -> targets -> (delegated targets).
 * Metadata can be loaded even if it is expired (or in the snapshot case if the
   meta info does not match): this is called "intermediate metadata".
 * Intermediate metadata can _only_ be used to load newer versions of the
   same metadata: As an example an expired root can be used to load a new root.
 * Metadata is loadable only if metadata before it in loading order is loaded
   (and is not intermediate): As an example timestamp can be loaded if a
   final (non-expired) root has been loaded.
 * Metadata is not loadable if any metadata after it in loading order has been
   loaded: As an example new roots cannot be loaded if timestamp is loaded.

Exceptions are raised if metadata fails to load in any way.

Example of loading root, timestamp and snapshot:

>>> # Load local root (RepositoryErrors here stop the update)
>>> with open(root_path, "rb") as f:
>>>     trusted_set = TrustedMetadataSet(f.read(), EnvelopeType.METADATA)
>>>
>>> # update root from remote until no more are available
>>> with download(Root.type, trusted_set.root.version + 1) as f:
>>>     trusted_set.update_root(f.read())
>>>
>>> # load local timestamp, then update from remote
>>> try:
>>>     with open(timestamp_path, "rb") as f:
>>>         trusted_set.update_timestamp(f.read())
>>> except (RepositoryError, OSError):
>>>     pass # failure to load a local file is ok
>>>
>>> with download(Timestamp.type) as f:
>>>     trusted_set.update_timestamp(f.read())
>>>
>>> # load local snapshot, then update from remote if needed
>>> try:
>>>     with open(snapshot_path, "rb") as f:
>>>         trusted_set.update_snapshot(f.read())
>>> except (RepositoryError, OSError):
>>>     # local snapshot is not valid, load from remote
>>>     # (RepositoryErrors here stop the update)
>>>     with download(Snapshot.type, version) as f:
>>>         trusted_set.update_snapshot(f.read())
"""

from __future__ import annotations

import logging
from collections import abc
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from tuf.api import exceptions
from tuf.api.dsse import SimpleEnvelope
from tuf.api.metadata import (
    Metadata,
    Root,
    Signed,
    Snapshot,
    T,
    Targets,
    Timestamp,
)
from tuf.ngclient.config import EnvelopeType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from securesystemslib.signer import Signature

logger = logging.getLogger(__name__)

Delegator = Root | Targets

# Audit rejection rule identifiers (kept local to avoid a circular import
# with tuf.ngclient.audit). The audit layer maps its public rule names onto
# the same strings.
RULE_ROLLBACK = "version_rollback"
RULE_EXPIRED = "metadata_expired"
RULE_HASH_MISMATCH = "hash_mismatch"
RULE_SIGNATURES = "insufficient_signatures"
RULE_ROLE_MISSING = "role_missing"
RULE_MALFORMED = "malformed_metadata"


@dataclass(frozen=True)
class AcceptedMetadata:
    """Evidence of a metadata role that passed verification.

    Carries everything an audit observer needs to describe the verification
    without re-running it: the deserialized payload, the exact bytes that were
    signed, the presented signatures, the delegator that verified them and
    (for root) the previous root version also used in verification.
    """

    role: str
    delegator_name: str
    payload: Signed
    payload_bytes: bytes
    signatures: dict[str, Signature]
    raw_data: bytes
    delegator: Delegator | None
    previous_root: Root | None
    source: str


@dataclass(frozen=True)
class RejectedMetadata:
    """Evidence of a metadata role that failed verification.

    ``raw_data`` may be ``None`` if the bytes could not even be downloaded.
    """

    role: str
    delegator_name: str
    raw_data: bytes | None
    rule: str
    error: Exception
    source: str


class AuditHook(Protocol):
    """Observability side-channel for the metadata verification workflow.

    Implementations record an evidence chain. The protocol is strictly
    observational: hooks must not raise and must not influence verification.
    """

    def metadata_accepted(self, event: AcceptedMetadata) -> None: ...

    def metadata_rejected(self, event: RejectedMetadata) -> None: ...


class TrustedMetadataSet(abc.Mapping):
    """Internal class to keep track of trusted metadata in ``Updater``.

    ``TrustedMetadataSet`` ensures that the collection of metadata in it is
    valid and trusted through the whole client update workflow. It provides
    easy ways to update the metadata with the caller making decisions on
    what is updated.
    """

    def __init__(
        self,
        root_data: bytes,
        envelope_type: EnvelopeType,
        audit_hook: AuditHook | None = None,
        audit_source: str = "remote",
    ):
        """Initialize ``TrustedMetadataSet`` by loading trusted root metadata.

        Args:
            root_data: Trusted root metadata as bytes. Note that this metadata
                will only be verified by itself: it is the source of trust for
                all metadata in the ``TrustedMetadataSet``
            envelope_type: Configures deserialization and verification mode of
                TUF metadata.
            audit_hook: Optional observability hook. When set, every
                successful or failed metadata verification is reported to it.
                The hook is strictly side-channel: it never affects the
                verification result.
            audit_source: Where ``root_data`` came from ("local" cache or
                "remote" repository). Used only for audit evidence.

        Raises:
            RepositoryError: Metadata failed to load or verify. The actual
                error type and content will contain more details.
        """
        self._trusted_set: dict[str, Signed] = {}
        self._trusted_delegations: set[tuple[str, str]] = set()
        self.reference_time = datetime.now(UTC)
        self._audit_hook = audit_hook
        # Where the currently verified bytes came from ("local" cache or
        # "remote" repository). The Updater sets this around each call.
        self.audit_source = audit_source

        if envelope_type is EnvelopeType.SIMPLE:
            self._load_data = _load_from_simple_envelope
        else:
            self._load_data = _load_from_metadata

        # Load and validate the local root metadata. Valid initial trusted root
        # metadata is required
        logger.debug("Updating initial trusted root")
        self._load_trusted_root(root_data)

    def _audit_accepted(
        self,
        role: str,
        delegator_name: str,
        payload: Signed,
        payload_bytes: bytes,
        signatures: dict[str, Signature],
        raw_data: bytes,
        delegator: Delegator | None,
        previous_root: Root | None,
    ) -> None:
        """Report a successful verification to the audit hook, if configured."""
        if self._audit_hook is None:
            return
        event = AcceptedMetadata(
            role=role,
            delegator_name=delegator_name,
            payload=payload,
            payload_bytes=payload_bytes,
            signatures=signatures,
            raw_data=raw_data,
            delegator=delegator,
            previous_root=previous_root,
            source=self.audit_source,
        )
        try:
            self._audit_hook.metadata_accepted(event)
        except Exception:  # noqa: BLE001
            logger.debug("audit hook failed", exc_info=True)

    def _audit_rejected(
        self,
        role: str,
        delegator_name: str,
        raw_data: bytes | None,
        rule: str,
        error: Exception,
    ) -> None:
        """Report a failed verification to the audit hook, if configured."""
        if self._audit_hook is None:
            return
        event = RejectedMetadata(
            role=role,
            delegator_name=delegator_name,
            raw_data=raw_data,
            rule=rule,
            error=error,
            source=self.audit_source,
        )
        try:
            self._audit_hook.metadata_rejected(event)
        except Exception:  # noqa: BLE001
            logger.debug("audit hook failed", exc_info=True)

    def contains(self, role: str, delegator: str) -> bool:
        """Check if ``role`` is in ``TrustedMetadataSet`` and was verified
        against ``delegator``.
        """
        return (delegator, role) in self._trusted_delegations

    def _add(self, role: str, signed: Signed, delegator: str) -> None:
        """Add role to trusted set, keep track of delegator(s)"""
        self._trusted_set[role] = signed
        self._trusted_delegations.add((delegator, role))

    def __getitem__(self, role: str) -> Signed:
        """Return current ``Signed`` for ``role``."""
        return self._trusted_set[role]

    def __len__(self) -> int:
        """Return number of ``Signed`` objects in ``TrustedMetadataSet``."""
        return len(self._trusted_set)

    def __iter__(self) -> Iterator[Signed]:
        """Return iterator over ``Signed`` objects in
        ``TrustedMetadataSet``.
        """
        return iter(self._trusted_set.values())

    # Helper properties for top level metadata
    @property
    def root(self) -> Root:
        """Get current root."""
        return cast("Root", self._trusted_set[Root.type])

    @property
    def timestamp(self) -> Timestamp:
        """Get current timestamp."""
        return cast("Timestamp", self._trusted_set[Timestamp.type])

    @property
    def snapshot(self) -> Snapshot:
        """Get current snapshot."""
        return cast("Snapshot", self._trusted_set[Snapshot.type])

    @property
    def targets(self) -> Targets:
        """Get current top-level targets."""
        return cast("Targets", self._trusted_set[Targets.type])

    # Methods for updating metadata
    def update_root(self, data: bytes) -> Root:
        """Verify and load ``data`` as new root metadata.

        Note that an expired intermediate root is considered valid: expiry is
        only checked for the final root in ``update_timestamp()``.

        Args:
            data: Unverified new root metadata as bytes

        Raises:
            RuntimeError: This function is called after updating timestamp.
            RepositoryError: Metadata failed to load or verify. The actual
                error type and content will contain more details.

        Returns:
            Deserialized and verified ``Root`` object
        """
        if Timestamp.type in self._trusted_set:
            raise RuntimeError("Cannot update root after timestamp")
        logger.debug("Updating root")

        previous_root = self.root
        try:
            new_root, new_root_bytes, new_root_signatures = self._load_data(
                Root, data, previous_root
            )
            if new_root.version != previous_root.version + 1:
                raise exceptions.BadVersionNumberError(
                    f"Expected root version {previous_root.version + 1}"
                    f" instead got version {new_root.version}"
                )

            # Verify that new root is signed by itself
            new_root.verify_delegate(
                Root.type, new_root_bytes, new_root_signatures
            )
        except exceptions.UnsignedMetadataError as e:
            self._audit_rejected(
                Root.type,
                Root.type,
                data,
                RULE_SIGNATURES,
                e,
            )
            raise
        except exceptions.BadVersionNumberError as e:
            self._audit_rejected(Root.type, Root.type, data, RULE_ROLLBACK, e)
            raise
        except exceptions.RepositoryError as e:
            self._audit_rejected(Root.type, Root.type, data, RULE_MALFORMED, e)
            raise

        self._add(Root.type, new_root, Root.type)
        logger.debug("Updated root v%d", new_root.version)
        self._audit_accepted(
            Root.type,
            Root.type,
            new_root,
            new_root_bytes,
            new_root_signatures,
            data,
            new_root,
            previous_root
            if previous_root.version != new_root.version
            else None,
        )

        return new_root

    def update_timestamp(self, data: bytes) -> Timestamp:
        """Verify and load ``data`` as new timestamp metadata.

        Note that an intermediate timestamp is allowed to be expired:
        ``TrustedMetadataSet`` will throw an ``ExpiredMetadataError`` in
        this case but the intermediate timestamp will be loaded. This way
        a newer timestamp can still be loaded (and the intermediate
        timestamp will be used for rollback protection). Expired timestamp
        will prevent loading snapshot metadata.

        Args:
            data: Unverified new timestamp metadata as bytes

        Raises:
            RuntimeError: This function is called after updating snapshot.
            RepositoryError: Metadata failed to load or verify as final
                timestamp. The actual error type and content will contain
                more details.

        Returns:
            Deserialized and verified ``Timestamp`` object
        """
        if Snapshot.type in self._trusted_set:
            raise RuntimeError("Cannot update timestamp after snapshot")

        # client workflow 5.3.10: Make sure final root is not expired.
        if self.root.is_expired(self.reference_time):
            expiry_error = exceptions.ExpiredMetadataError(
                "Final root.json is expired"
            )
            self._audit_rejected(
                Timestamp.type,
                Root.type,
                data,
                RULE_EXPIRED,
                expiry_error,
            )
            raise expiry_error
        # No need to check for 5.3.11 (fast forward attack recovery):
        # timestamp/snapshot can not yet be loaded at this point

        try:
            new_timestamp, new_bytes, new_signatures = self._load_data(
                Timestamp, data, self.root
            )
        except exceptions.UnsignedMetadataError as e:
            self._audit_rejected(
                Timestamp.type, Root.type, data, RULE_SIGNATURES, e
            )
            raise
        except exceptions.RepositoryError as e:
            self._audit_rejected(
                Timestamp.type, Root.type, data, RULE_MALFORMED, e
            )
            raise

        # If an existing trusted timestamp is updated,
        # check for a rollback attack
        if Timestamp.type in self._trusted_set:
            # Prevent rolling back timestamp version
            if new_timestamp.version < self.timestamp.version:
                rollback_error = exceptions.BadVersionNumberError(
                    f"New timestamp version {new_timestamp.version} must"
                    f" be >= {self.timestamp.version}"
                )
                self._audit_rejected(
                    Timestamp.type,
                    Root.type,
                    data,
                    RULE_ROLLBACK,
                    rollback_error,
                )
                raise rollback_error
            # Keep using old timestamp if versions are equal.
            if new_timestamp.version == self.timestamp.version:
                # Equal version is a normal condition, not a rejection.
                raise exceptions.EqualVersionNumberError

            # Prevent rolling back snapshot version
            snapshot_meta = self.timestamp.snapshot_meta
            new_snapshot_meta = new_timestamp.snapshot_meta
            if new_snapshot_meta.version < snapshot_meta.version:
                snapshot_rollback_error = exceptions.BadVersionNumberError(
                    f"New snapshot version must be >= {snapshot_meta.version}"
                    f", got version {new_snapshot_meta.version}"
                )
                self._audit_rejected(
                    Timestamp.type,
                    Root.type,
                    data,
                    RULE_ROLLBACK,
                    snapshot_rollback_error,
                )
                raise snapshot_rollback_error

        # expiry not checked to allow old timestamp to be used for rollback
        # protection of new timestamp: expiry is checked in update_snapshot()

        self._add(Timestamp.type, new_timestamp, Root.type)
        logger.debug("Updated timestamp v%d", new_timestamp.version)

        # timestamp is loaded: raise if it is not valid _final_ timestamp
        try:
            self._check_final_timestamp()
        except exceptions.ExpiredMetadataError as e:
            self._audit_rejected(
                Timestamp.type, Root.type, data, RULE_EXPIRED, e
            )
            raise

        self._audit_accepted(
            Timestamp.type,
            Root.type,
            new_timestamp,
            new_bytes,
            new_signatures,
            data,
            self.root,
            None,
        )
        return new_timestamp

    def _check_final_timestamp(self) -> None:
        """Raise if timestamp is expired."""

        if self.timestamp.is_expired(self.reference_time):
            raise exceptions.ExpiredMetadataError("timestamp.json is expired")

    def update_snapshot(
        self, data: bytes, trusted: bool | None = False
    ) -> Snapshot:
        """Verify and load ``data`` as new snapshot metadata.

        Note that an intermediate snapshot is allowed to be expired and version
        is allowed to not match timestamp meta version: ``TrustedMetadataSet``
        will throw an ``ExpiredMetadataError``/``BadVersionNumberError`` in
        these cases but the intermediate snapshot will be loaded. This way a
        newer snapshot can still be loaded (and the intermediate snapshot will
        be used for rollback protection). Expired snapshot or snapshot that
        does not match timestamp meta version will prevent loading targets.

        Args:
            data: Unverified new snapshot metadata as bytes
            trusted: ``True`` if data has at some point been verified by
                ``TrustedMetadataSet`` as a valid snapshot. Purpose of trusted
                is to allow loading of locally stored snapshot as intermediate
                snapshot even if hashes in current timestamp meta no longer
                match data. Default is False.

        Raises:
            RuntimeError: This function is called before updating timestamp
                or after updating targets.
            RepositoryError: Data failed to load or verify as final snapshot.
                The actual error type and content will contain more details.

        Returns:
            Deserialized and verified ``Snapshot`` object
        """

        if Timestamp.type not in self._trusted_set:
            raise RuntimeError("Cannot update snapshot before timestamp")
        if Targets.type in self._trusted_set:
            raise RuntimeError("Cannot update snapshot after targets")
        logger.debug("Updating snapshot")

        # Snapshot cannot be loaded if final timestamp is expired
        try:
            self._check_final_timestamp()
        except exceptions.ExpiredMetadataError as e:
            self._audit_rejected(
                Snapshot.type, Root.type, data, RULE_EXPIRED, e
            )
            raise

        snapshot_meta = self.timestamp.snapshot_meta

        # Verify non-trusted data against the hashes in timestamp, if any.
        # Trusted snapshot data has already been verified once.
        if not trusted:
            try:
                snapshot_meta.verify_length_and_hashes(data)
            except exceptions.LengthOrHashMismatchError as e:
                self._audit_rejected(
                    Snapshot.type, Root.type, data, RULE_HASH_MISMATCH, e
                )
                raise

        try:
            new_snapshot, new_bytes, new_signatures = self._load_data(
                Snapshot, data, self.root
            )
        except exceptions.UnsignedMetadataError as e:
            self._audit_rejected(
                Snapshot.type, Root.type, data, RULE_SIGNATURES, e
            )
            raise
        except exceptions.RepositoryError as e:
            self._audit_rejected(
                Snapshot.type, Root.type, data, RULE_MALFORMED, e
            )
            raise

        # version not checked against meta version to allow old snapshot to be
        # used in rollback protection: it is checked when targets is updated

        # If an existing trusted snapshot is updated, check for rollback attack
        if Snapshot.type in self._trusted_set:
            self._check_snapshot_rollback(new_snapshot, data)

        # expiry not checked to allow old snapshot to be used for rollback
        # protection of new snapshot: it is checked when targets is updated

        self._add(Snapshot.type, new_snapshot, Root.type)
        logger.debug("Updated snapshot v%d", new_snapshot.version)

        # snapshot is loaded, but we raise if it's not valid _final_ snapshot
        self._audit_or_raise(
            Snapshot.type,
            Root.type,
            data,
            self._check_final_snapshot,
            {
                exceptions.ExpiredMetadataError: RULE_EXPIRED,
                exceptions.BadVersionNumberError: RULE_ROLLBACK,
            },
        )

        self._audit_accepted(
            Snapshot.type,
            Root.type,
            new_snapshot,
            new_bytes,
            new_signatures,
            data,
            self.root,
            None,
        )
        return new_snapshot

    def _check_snapshot_rollback(
        self, new_snapshot: Snapshot, data: bytes
    ) -> None:
        """Reject a snapshot that removes roles or rolls back their versions."""
        for filename, fileinfo in self.snapshot.meta.items():
            new_fileinfo = new_snapshot.meta.get(filename)

            # Prevent removal of any metadata in meta
            if new_fileinfo is None:
                error = exceptions.RepositoryError(
                    f"New snapshot is missing info for '{filename}'"
                )
                self._audit_rejected(
                    Snapshot.type,
                    Root.type,
                    data,
                    RULE_ROLE_MISSING,
                    error,
                )
                raise error

            # Prevent rollback of any metadata versions
            if new_fileinfo.version < fileinfo.version:
                error = exceptions.BadVersionNumberError(
                    f"Expected {filename} version "
                    f"{new_fileinfo.version}, got {fileinfo.version}."
                )
                self._audit_rejected(
                    Snapshot.type, Root.type, data, RULE_ROLLBACK, error
                )
                raise error

    def _audit_or_raise(
        self,
        role: str,
        delegator_name: str,
        data: bytes,
        check: abc.Callable[[], None],
        rules: abc.Mapping[type[Exception], str],
    ) -> None:
        """Run a validation check, auditing then re-raising mapped errors."""
        try:
            check()
        except tuple(rules) as e:
            self._audit_rejected(role, delegator_name, data, rules[type(e)], e)
            raise

    def _check_final_snapshot(self) -> None:
        """Raise if snapshot is expired or meta version does not match."""

        if self.snapshot.is_expired(self.reference_time):
            raise exceptions.ExpiredMetadataError("snapshot.json is expired")
        snapshot_meta = self.timestamp.snapshot_meta
        if self.snapshot.version != snapshot_meta.version:
            raise exceptions.BadVersionNumberError(
                f"Expected snapshot version {snapshot_meta.version}, "
                f"got {self.snapshot.version}"
            )

    def update_targets(self, data: bytes) -> Targets:
        """Verify and load ``data`` as new top-level targets metadata.

        Args:
            data: Unverified new targets metadata as bytes

        Raises:
            RepositoryError: Metadata failed to load or verify. The actual
                error type and content will contain more details.

        Returns:
            Deserialized and verified `Targets`` object
        """
        return self.update_delegated_targets(data, Targets.type, Root.type)

    def update_delegated_targets(
        self, data: bytes, role_name: str, delegator_name: str
    ) -> Targets:
        """Verify and load ``data`` as new metadata for target ``role_name``.

        Args:
            data: Unverified new metadata as bytes
            role_name: Role name of the new metadata
            delegator_name: Name of the role delegating to the new metadata

        Raises:
            RuntimeError: This function is called before updating snapshot.
            RepositoryError: Metadata failed to load or verify. The actual
                error type and content will contain more details.

        Returns:
            Deserialized and verified ``Targets`` object
        """
        if Snapshot.type not in self._trusted_set:
            raise RuntimeError("Cannot load targets before snapshot")

        # Targets cannot be loaded if final snapshot is expired or its version
        # does not match meta version in timestamp
        try:
            self._check_final_snapshot()
        except (
            exceptions.ExpiredMetadataError,
            exceptions.BadVersionNumberError,
        ) as e:
            rule = (
                RULE_EXPIRED
                if isinstance(e, exceptions.ExpiredMetadataError)
                else RULE_ROLLBACK
            )
            self._audit_rejected(role_name, delegator_name, data, rule, e)
            raise

        delegator: Delegator | None = self.get(delegator_name)
        if delegator is None:
            raise RuntimeError("Cannot load targets before delegator")

        logger.debug("Updating %s delegated by %s", role_name, delegator_name)

        # Verify against the hashes in snapshot, if any
        meta = self.snapshot.meta.get(f"{role_name}.json")
        if meta is None:
            error = exceptions.RepositoryError(
                f"Snapshot does not contain information for '{role_name}'"
            )
            self._audit_rejected(
                role_name, delegator_name, data, RULE_ROLE_MISSING, error
            )
            raise error

        try:
            meta.verify_length_and_hashes(data)
        except exceptions.LengthOrHashMismatchError as e:
            self._audit_rejected(
                role_name, delegator_name, data, RULE_HASH_MISMATCH, e
            )
            raise

        try:
            new_delegate, new_bytes, new_signatures = self._load_data(
                Targets, data, delegator, role_name
            )
        except exceptions.UnsignedMetadataError as e:
            self._audit_rejected(
                role_name, delegator_name, data, RULE_SIGNATURES, e
            )
            raise
        except exceptions.RepositoryError as e:
            self._audit_rejected(
                role_name, delegator_name, data, RULE_MALFORMED, e
            )
            raise

        version = new_delegate.version
        if version != meta.version:
            error = exceptions.BadVersionNumberError(
                f"Expected {role_name} v{meta.version}, got v{version}."
            )
            self._audit_rejected(
                role_name, delegator_name, data, RULE_ROLLBACK, error
            )
            raise error

        if new_delegate.is_expired(self.reference_time):
            error = exceptions.ExpiredMetadataError(
                f"New {role_name} is expired"
            )
            self._audit_rejected(
                role_name, delegator_name, data, RULE_EXPIRED, error
            )
            raise error

        self._add(role_name, new_delegate, delegator_name)
        logger.debug("Updated %s v%d", role_name, version)
        self._audit_accepted(
            role_name,
            delegator_name,
            new_delegate,
            new_bytes,
            new_signatures,
            data,
            delegator,
            None,
        )

        return new_delegate

    def _load_trusted_root(self, data: bytes) -> None:
        """Verify and load ``data`` as trusted root metadata.

        Note that an expired initial root is considered valid: expiry is
        only checked for the final root in ``update_timestamp()``.
        """
        try:
            new_root, new_root_bytes, new_root_signatures = self._load_data(
                Root, data
            )
            new_root.verify_delegate(
                Root.type, new_root_bytes, new_root_signatures
            )
        except exceptions.UnsignedMetadataError as e:
            self._audit_rejected(Root.type, Root.type, data, RULE_SIGNATURES, e)
            raise
        except exceptions.RepositoryError as e:
            self._audit_rejected(Root.type, Root.type, data, RULE_MALFORMED, e)
            raise

        self._add(Root.type, new_root, Root.type)
        logger.debug("Loaded trusted root v%d", new_root.version)
        self._audit_accepted(
            Root.type,
            Root.type,
            new_root,
            new_root_bytes,
            new_root_signatures,
            data,
            new_root,
            None,
        )


def _load_from_metadata(
    role: type[T],
    data: bytes,
    delegator: Delegator | None = None,
    role_name: str | None = None,
) -> tuple[T, bytes, dict[str, Signature]]:
    """Load traditional metadata bytes, and extract and verify payload.

    If no delegator is passed, verification is skipped. Returns a tuple of
    deserialized payload, signed payload bytes, and signatures.
    """
    md = Metadata[T].from_bytes(data)

    if md.signed.type != role.type:
        raise exceptions.RepositoryError(
            f"Expected '{role.type}', got '{md.signed.type}'"
        )

    if delegator:
        if role_name is None:
            role_name = role.type

        delegator.verify_delegate(role_name, md.signed_bytes, md.signatures)

    return md.signed, md.signed_bytes, md.signatures


def _load_from_simple_envelope(
    role: type[T],
    data: bytes,
    delegator: Delegator | None = None,
    role_name: str | None = None,
) -> tuple[T, bytes, dict[str, Signature]]:
    """Load simple envelope bytes, and extract and verify payload.

    If no delegator is passed, verification is skipped. Returns a tuple of
    deserialized payload, signed payload bytes, and signatures.
    """

    envelope = SimpleEnvelope[T].from_bytes(data)

    if envelope.payload_type != SimpleEnvelope.DEFAULT_PAYLOAD_TYPE:
        raise exceptions.RepositoryError(
            f"Expected '{SimpleEnvelope.DEFAULT_PAYLOAD_TYPE}', "
            f"got '{envelope.payload_type}'"
        )

    if delegator:
        if role_name is None:
            role_name = role.type
        delegator.verify_delegate(
            role_name, envelope.pae(), envelope.signatures
        )

    signed = envelope.get_signed()
    if signed.type != role.type:
        raise exceptions.RepositoryError(
            f"Expected '{role.type}', got '{signed.type}'"
        )

    return signed, envelope.pae(), envelope.signatures
