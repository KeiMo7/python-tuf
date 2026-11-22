# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Configuration options for ``Updater`` class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, unique


@unique
class EnvelopeType(Flag):
    """Configures deserialization and verification mode of TUF metadata.

    Args:
        METADATA: Traditional canonical JSON -based TUF Metadata.
        SIMPLE: Dead Simple Signing Envelope. (experimental)
    """

    METADATA = 1
    SIMPLE = 2


@dataclass(frozen=True)
class UpdaterConfig:
    """Used to store ``Updater`` configuration.

    Configuration is immutable. Set options in the constructor or use
    :func:`dataclasses.replace` to create a modified configuration for a new
    ``Updater`` instance.

    Args:
        max_root_rotations: Maximum number of root rotations.
        max_delegations: Maximum number of delegations.
        root_max_length: Maximum length of a root metadata file.
        timestamp_max_length: Maximum length of a timestamp metadata file.
        snapshot_max_length: Maximum length of a snapshot metadata file.
        targets_max_length: Maximum length of a targets metadata file.
        prefix_targets_with_hash: When `consistent snapshots
            <https://theupdateframework.github.io/specification/latest/#consistent-snapshots>`_
            are used, target download URLs are formed by prefixing the filename
            with a hash digest of file content by default. This can be
            overridden by setting ``prefix_targets_with_hash`` to ``False``.
        envelope_type: Configures deserialization and verification mode of TUF
            metadata. Per default, it is treated as traditional canonical JSON
            -based TUF Metadata.
        app_user_agent: Application user agent, e.g. "MyApp/1.0.0". This will be
            prefixed to ngclient user agent when the default fetcher is used.
        lock_timeout: Maximum time in seconds to wait for exclusive access to
            the local metadata directory. If the lock cannot be acquired in
            time, ``LockTimeoutError`` is raised: the updater never writes
            local metadata without holding the lock.
        lease_timeout: Maximum time in seconds a lock may be considered held by
            a different process without that process making progress. The lock
            is backed by the operating system and released automatically when
            the holding process dies, so this only guards against a live process
            that has stopped making progress. ``None`` disables lease based
            recovery.
        generations_to_keep: Number of committed metadata generations to keep on
            disk for recovery and rollback. The visible generation is never
            removed.
    """

    max_root_rotations: int = 256
    max_delegations: int = 32
    root_max_length: int = 512000  # bytes
    timestamp_max_length: int = 16384  # bytes
    snapshot_max_length: int = 2000000  # bytes
    targets_max_length: int = 5000000  # bytes
    prefix_targets_with_hash: bool = True
    envelope_type: EnvelopeType = EnvelopeType.METADATA
    app_user_agent: str | None = None
    lock_timeout: float = 30.0
    lease_timeout: float | None = 300.0
    generations_to_keep: int = 5
