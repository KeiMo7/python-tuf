# Copyright 2020, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Client update workflow implementation.

The ``Updater`` class provides an implementation of the
`TUF client workflow
<https://theupdateframework.github.io/specification/latest/#detailed-client-workflow>`_.
``Updater`` provides an API to query available targets and to download them in a
secure manner: All downloaded files are verified by signed metadata.

High-level description of ``Updater`` functionality:
  * Initializing an ``Updater`` loads and validates the trusted local root
    metadata: This root metadata is used as the source of trust for all other
    metadata. Updater should always be initialized with the ``bootstrap``
    argument: pass ``bootstrap=None`` only to explicitly opt into using the
    cached root.json as the trust anchor.
  * ``refresh()`` can optionally be called to update and load all top-level
    metadata as described in the specification, using both locally cached
    metadata and metadata downloaded from the remote repository. If refresh is
    not done explicitly, it will happen automatically during the first target
    info lookup.
  * ``Updater`` can be used to download targets. For each target:

      * ``Updater.get_targetinfo()`` is first used to find information about a
        specific target. This will load new targets metadata as needed (from
        local cache or remote repository).
      * ``Updater.find_cached_target()`` can optionally be used to check if a
        target file is already locally cached.
      * ``Updater.download_target()`` downloads a target file and ensures it is
        verified correct by the metadata.

Note that applications using ``Updater`` may run multiple instances (processes
or threads) against the same metadata and target directories. Local metadata is
kept in crash-safe, atomically committed generations (see
``tuf.ngclient._internal.local_store``): readers always see one complete
verified set, writers serialize on an OS-backed lock, and a process that is
killed mid-write can never leave partially updated metadata visible. Local
target files are written to a temporary file, verified, and only then renamed
into place.

A simple example of using the Updater to implement a Python TUF client that
downloads target files is available in `examples/client
<https://github.com/theupdateframework/python-tuf/tree/develop/examples/client>`_.

Notes on how Updater uses HTTP by default:
  * urllib3 is the HTTP library
  * Typically all requests are retried by urllib3 three times (in cases where
    this seems useful)
  * Operating system certificate store is used for TLS, in other words
    ``certifi`` is not used as the certificate source
  * Proxy use can be configured with ``https_proxy`` and other similar
    environment variables

All of the HTTP decisions can be changed with ``fetcher`` argument:
Custom ``FetcherInterface`` implementations are possible. The alternative
``RequestsFetcher`` implementation is also provided (although deprecated).
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib import parse

from tuf.api import exceptions
from tuf.api.metadata import Root, Snapshot, TargetFile, Targets, Timestamp
from tuf.ngclient._internal.local_store import Generation, MetadataStore
from tuf.ngclient._internal.trusted_metadata_set import TrustedMetadataSet
from tuf.ngclient.config import EnvelopeType, UpdaterConfig
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

if TYPE_CHECKING:
    from tuf.ngclient.fetcher import FetcherInterface

logger = logging.getLogger(__name__)


class Updater:
    """Creates a new ``Updater`` instance and loads trusted root metadata.

    Args:
        metadata_dir: Local metadata directory. Directory must be
            writable. If ``bootstrap`` is ``None``, this directory must contain
            a trusted root.json file.
        metadata_base_url: Base URL for all remote metadata downloads
        target_dir: Local targets directory. Directory must be writable. It
            will be used as the default target download directory by
            ``find_cached_target()`` and ``download_target()``
        target_base_url: ``Optional``; Default base URL for all remote target
            downloads. Can be individually set in ``download_target()``
        fetcher: ``Optional``; ``FetcherInterface`` implementation used to
            download both metadata and targets. Default is ``Urllib3Fetcher``
        config: ``Optional``; ``UpdaterConfig`` could be used to setup common
            configuration options. Configuration cannot be changed after
            initialization: create a new ``Updater`` to use different options.
        bootstrap: Initial root metadata bytes. This argument is required.
            Pass the embedded root metadata bytes for secure initialization.
            Pass ``None`` only if you explicitly want to use the cached
            root.json as the trust anchor (not recommended for most
            deployments).

    Raises:
        OSError: Local root.json cannot be read
        RepositoryError: Local root.json is invalid
    """

    def __init__(
        self,
        metadata_dir: str,
        metadata_base_url: str,
        target_dir: str | None = None,
        target_base_url: str | None = None,
        fetcher: FetcherInterface | None = None,
        config: UpdaterConfig | None = None,
        *,
        bootstrap: bytes | None,
    ):
        self._dir = metadata_dir
        self._metadata_base_url = _ensure_trailing_slash(metadata_base_url)
        self.target_dir = target_dir
        if target_base_url is None:
            self._target_base_url = None
        else:
            self._target_base_url = _ensure_trailing_slash(target_base_url)

        self._config = config or UpdaterConfig()
        if fetcher is not None:
            self._fetcher = fetcher
        else:
            self._fetcher = Urllib3Fetcher(
                app_user_agent=self.config.app_user_agent
            )
        supported_envelopes = [EnvelopeType.METADATA, EnvelopeType.SIMPLE]
        if self.config.envelope_type not in supported_envelopes:
            raise ValueError(
                f"config: envelope_type must be one of {supported_envelopes}, "
                f"got '{self.config.envelope_type}'"
            )

        self._store = MetadataStore(
            metadata_dir,
            lock_timeout=self.config.lock_timeout,
            lease_timeout=self.config.lease_timeout,
            generations_to_keep=self.config.generations_to_keep,
        )
        self._local_generation: Generation | None = None
        self._pending_metadata: dict[str, bytes] = {}
        self._pending_versions: dict[str, int] = {}
        self._pending_root: tuple[int, bytes] | None = None
        self._allow_rollback = False

        if bootstrap is None:
            # if no root was provided, use the cached non-versioned root.json
            bootstrap = self._load_local_metadata(Root.type)

        # Load the initial root, make sure it's cached
        self._trusted_set = TrustedMetadataSet(
            bootstrap, self.config.envelope_type
        )
        self._persist_root(self._trusted_set.root.version, bootstrap)
        self._update_root_symlink()

    @property
    def config(self) -> UpdaterConfig:
        """Read-only configuration used for this Updater's lifetime."""
        return self._config

    def refresh(self) -> None:
        """Refresh top-level metadata.

        Downloads, verifies, and loads metadata for the top-level roles in the
        specified order (root -> timestamp -> snapshot -> targets) implementing
        all the checks required in the TUF client workflow.

        Metadata is committed locally as one atomic generation: readers never
        see a mixture of old and new metadata, even if this process is killed.
        If another process is refreshing at the same time, the verified result
        of whichever process finishes last is committed without ever going
        backwards in version.

        A ``refresh()`` can be done only once during the lifetime of an Updater.
        If ``refresh()`` has not been explicitly called before the first
        ``get_targetinfo()`` call, it will be done implicitly at that time.

        The metadata for delegated roles is not updated by ``refresh()``:
        that happens on demand during ``get_targetinfo()``. However, if the
        repository uses `consistent_snapshot
        <https://theupdateframework.github.io/specification/latest/#consistent-snapshots>`_,
        then all metadata downloaded by the Updater will use the same consistent
        repository state.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way
            LockTimeoutError: Exclusive access to the metadata directory could
                not be acquired in time
        """

        self._load_root()
        error: BaseException | None = None
        try:
            self._load_timestamp()
            self._load_snapshot()
            self._load_targets(Targets.type, Root.type)
        except BaseException as e:
            error = e
            raise
        finally:
            # Verified metadata that was already downloaded is part of the
            # locally trusted set even if a later role failed: commit it as one
            # atomic generation, exactly like a fully successful refresh.
            try:
                with self._store.lock():
                    # Another process may have committed while this Updater was
                    # downloading: merge into the freshest visible generation
                    # so that newer verified metadata is never replaced by
                    # older.
                    self._store.invalidate()
                    self._local_generation = self._store.load()
                    self._commit_local_metadata()
            except (
                exceptions.LockTimeoutError,
                exceptions.IncompleteMetadataGenerationError,
            ) as e:
                if error is None:
                    raise
                # The refresh already failed: report the commit failure but do
                # not hide the original verification error.
                logger.error("Could not commit verified local metadata: %s", e)

        logger.info(
            "Trusted local metadata: %s", self.local_metadata_versions()
        )

    def _generate_target_file_path(self, targetinfo: TargetFile) -> str:
        if self.target_dir is None:
            raise ValueError("target_dir must be set if filepath is not given")

        # Use URL encoded target path as filename
        filename = parse.quote(targetinfo.path, "")
        return os.path.join(self.target_dir, filename)

    def get_targetinfo(self, target_path: str) -> TargetFile | None:
        """Return ``TargetFile`` instance with information for ``target_path``.

        The return value can be used as an argument to
        ``download_target()`` and ``find_cached_target()``.

        If ``refresh()`` has not been called before calling
        ``get_targetinfo()``, the refresh will be done implicitly.

        As a side-effect this method downloads all the additional (delegated
        targets) metadata it needs to return the target information.

        Args:
            target_path: `path-relative-URL string
                <https://url.spec.whatwg.org/#path-relative-url-string>`_
                that uniquely identifies the target within the repository.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way

        Returns:
            ``TargetFile`` instance or ``None``.
        """

        if Targets.type not in self._trusted_set:
            self.refresh()
        return self._preorder_depth_first_walk(target_path)

    def find_cached_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
    ) -> str | None:
        """Check whether a local file is an up to date target.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to file. If ``None``, a file path is
                generated based on ``target_dir`` constructor argument.

        Raises:
            ValueError: Incorrect arguments

        Returns:
            Local file path if the file is an up to date target file.
            ``None`` if file is not found or it is not up to date.
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)

        try:
            with open(filepath, "rb") as target_file:
                targetinfo.verify_length_and_hashes(target_file)
            return filepath
        except (OSError, exceptions.LengthOrHashMismatchError):
            return None

    def download_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
        target_base_url: str | None = None,
    ) -> str:
        """Download the target file specified by ``targetinfo``.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to download into. If ``None``, the file is
                downloaded into directory defined by ``target_dir`` constructor
                argument using a generated filename. If file already exists,
                it is overwritten.
            target_base_url: Base URL used to form the final target
                download URL. Default is the value provided in ``Updater()``

        Raises:
            ValueError: Invalid arguments
            DownloadError: Download of the target file failed in some way
            RepositoryError: Downloaded target failed to be verified in some way
            OSError: Failed to write target to file

        Returns:
            Local path to downloaded file
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)
        destination_dir = os.path.dirname(filepath) or "."
        Path(destination_dir).mkdir(exist_ok=True, parents=True)

        if target_base_url is None:
            if self._target_base_url is None:
                raise ValueError(
                    "target_base_url must be set in either "
                    "download_target() or constructor"
                )

            target_base_url = self._target_base_url
        else:
            target_base_url = _ensure_trailing_slash(target_base_url)

        target_filepath = targetinfo.path
        consistent_snapshot = self._trusted_set.root.consistent_snapshot
        if consistent_snapshot and self.config.prefix_targets_with_hash:
            hashes = list(targetinfo.hashes.values())
            dirname, sep, basename = target_filepath.rpartition("/")
            target_filepath = f"{dirname}{sep}{hashes[0]}.{basename}"
        full_url = f"{target_base_url}{target_filepath}"

        # Write to a temporary file next to the destination and only rename it
        # into place after verification: a killed process can never leave a
        # partial target at the visible path, and a concurrent reader either
        # sees no file or a complete, verified one.
        fd, temp_path = tempfile.mkstemp(
            dir=destination_dir, prefix=".download-", suffix=".tmp"
        )
        try:
            with (
                self._fetcher.download_file(
                    full_url, targetinfo.length
                ) as target_file,
                os.fdopen(fd, "wb") as destination_file,
            ):
                targetinfo.verify_length_and_hashes(target_file)

                target_file.seek(0)
                shutil.copyfileobj(target_file, destination_file)
                destination_file.flush()
                os.fsync(destination_file.fileno())
            # Re-verify the bytes that are actually going to be published.
            with open(temp_path, "rb") as written_file:
                targetinfo.verify_length_and_hashes(written_file)
            os.replace(temp_path, filepath)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_path)
            raise

        logger.debug("Downloaded target %s", targetinfo.path)
        return filepath

    def _download_metadata(
        self, rolename: str, length: int, version: int | None = None
    ) -> bytes:
        """Download a metadata file and return it as bytes."""
        encoded_name = parse.quote(rolename, "")
        if version is None:
            url = f"{self._metadata_base_url}{encoded_name}.json"
        else:
            url = f"{self._metadata_base_url}{version}.{encoded_name}.json"
        return self._fetcher.download_bytes(url, length)

    def _load_local_metadata(self, rolename: str) -> bytes:
        """Read local metadata from the current committed generation.

        The generation is resolved once per Updater and cached, so all reads in
        one client workflow come from the same complete metadata set.
        """
        if rolename == Root.type:
            # Root is an upgrade chain: the newest verified root in the local
            # generation is the trust anchor, even if an individual root in the
            # chain was rejected during an earlier, interrupted update.
            return self._newest_local_root()
        return self._store.read(rolename)

    def _newest_local_root(self) -> bytes:
        """Return the newest root stored in the local generation.

        Falls back to the ``root.json`` anchor when no generation exists (for
        example a metadata directory prepared by an older version).
        """
        try:
            generation = self._store.load()
        except (OSError, exceptions.RepositoryError):
            generation = None
        if generation is not None and Root.type in generation.roles:
            return generation.read(Root.type)
        return self._load_local_root()

    def _load_local_root(self) -> bytes:
        """Read the local root anchor (``root.json``).

        The anchor is the independent, atomically updated record of the trusted
        root; it also exists inside the committed generation.
        """
        with open(os.path.join(self._dir, "root.json"), "rb") as f:
            return f.read()

    def _persist_metadata(self, rolename: str, data: bytes) -> None:
        """Store verified metadata for ``rolename`` in the next generation.

        The write only becomes visible when :meth:`_commit_local_metadata` runs,
        so a set of metadata that is refreshed together is also committed
        together.
        """
        self._pending_metadata[rolename] = data
        self._pending_versions[rolename] = _metadata_version(data)

    def _commit_local_metadata(self) -> None:
        """Atomically make all verified metadata visible. Lock must be held.

        A committed generation is always internally consistent: it is the set
        of metadata this Updater has verified from one trusted root, including
        the delegated roles it has loaded so far.
        """
        if not self._pending_metadata and self._pending_root is None:
            return
        # Use the freshest visible generation as the base: roles that were not
        # updated in this refresh are carried over, so the new generation is
        # always a complete set. ``load()`` is cached and recovery has already
        # run while the lock was held.
        self._local_generation = self._store.load()
        pending = dict(self._pending_metadata)
        versions = dict(self._pending_versions)
        root = self._pending_root
        # Never replace a newer locally committed root with an older one
        # (the bootstrap root, when the local root chain is ahead of it).
        if (
            root is not None
            and self._local_generation is not None
            and self._local_generation.versions.get(Root.type, 0) > root[0]
        ):
            root = None
        generation = self._store.commit(
            self._local_generation,
            pending,
            versions,
            root=root,
            allow_rollback=self._allow_rollback,
        )
        self._local_generation = generation
        if root is not None and self._pending_root == root:
            self._pending_root = None
        # Drop only what was actually committed: a delegated role may have
        # been queued while a top-level commit was in progress.
        for role, data in pending.items():
            if self._pending_metadata.get(role) == data:
                del self._pending_metadata[role]
                self._pending_versions.pop(role, None)
        # A rollback is only safe for the refresh that observed the verified
        # root rotation: later commits use normal monotonic checks again.
        self._allow_rollback = False

    def _persist_root(self, version: int, data: bytes) -> None:
        """Store verified root metadata in the append-only root history.

        The root chain is kept independently of the metadata generations so a
        newer root can always be verified, and so the anchor can point at any
        historically verified root. The root is also recorded in the next
        committed generation (see :meth:`_commit_local_metadata`).
        """
        rootdir = Path(self._dir, "root_history")
        rootdir.mkdir(exist_ok=True, parents=True)
        self._persist_file(str(rootdir / f"{version}.root.json"), data)
        self._pending_root = (version, data)

    def _persist_file(self, filename: str, data: bytes) -> None:
        """Write a file to disk atomically to avoid data loss."""
        temp_file_name = None
        # The temporary file must live on the same filesystem as the target,
        # so create it next to the target rather than in the metadata root.
        directory = os.path.dirname(filename) or "."
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_file.name, filename)
        except OSError as e:
            # remove tempfile if we managed to create one,
            # then let the exception happen
            if temp_file_name is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(temp_file_name)
            raise e

    def _update_root_symlink(self) -> None:
        """Point root.json at the current trusted root version.

        The anchor is replaced with an atomic rename (and root history entries
        are written atomically), so a concurrent reader never observes a
        missing or half-replaced anchor. No metadata lock is needed: this is
        not part of the generation commit, and the anchor always points at an
        immutable, individually verifiable root file.
        """
        version = self._trusted_set.root.version
        current = os.path.join("root_history", f"{version}.root.json")
        linkname = os.path.join(self._dir, "root.json")
        self._replace_root_anchor(linkname, current)

    def local_metadata_versions(self) -> dict[str, int]:
        """Return the versions of the locally committed metadata.

        This is the record of "which metadata version this client trusts": it
        can be logged to make a later failure investigation possible, and it
        can be compared between processes to assert that they agree.

        Returns:
            Mapping of role name to version for the visible local generation.
            Root is included when the generation records it.
        """
        generation = self._store.load()
        if generation is None:
            return {Root.type: self._trusted_set.root.version}
        return generation.versions

    def _replace_root_anchor(self, linkname: str, current: str) -> None:
        """Atomically replace ``linkname`` with a link to ``current``.

        The new link is created under a temporary name and renamed over the old
        one, so readers never observe a missing anchor. If replacing fails the
        previous anchor is kept and the error is raised.
        """
        source = os.path.join(self._dir, current)
        temp_path = os.path.join(self._dir, f".root-{uuid.uuid4().hex}.tmp")
        try:
            try:
                os.symlink(current, temp_path)
            except OSError as e:
                if getattr(e, "winerror", None) == 1314:
                    # Fallback for NTFS "required privilege is not held by
                    # client": link (or copy) the immutable root file instead
                    os.link(source, temp_path)
                else:
                    raise
            os.replace(temp_path, linkname)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_path)
            raise

    def _load_root(self) -> None:
        """Load root metadata.

        Sequentially load newer root metadata versions. First try to load from
        local cache and if that does not work, from the remote repository.

        If metadata is loaded from remote repository, store it in local cache.
        """

        # Update the root role
        lower_bound = self._trusted_set.root.version + 1
        upper_bound = lower_bound + self.config.max_root_rotations

        try:
            for next_version in range(lower_bound, upper_bound):
                # look for next_version in local cache
                try:
                    root_path = os.path.join(
                        self._dir, "root_history", f"{next_version}.root.json"
                    )
                    with open(root_path, "rb") as f:
                        self._trusted_set.update_root(f.read())
                    continue
                except (OSError, exceptions.RepositoryError) as e:
                    # this root did not exist locally or is invalid
                    logger.debug("Local root is not valid: %s", e)

                # next_version was not found locally, try remote
                try:
                    data = self._download_metadata(
                        Root.type,
                        self.config.root_max_length,
                        next_version,
                    )
                    self._trusted_set.update_root(data)
                    self._persist_root(next_version, data)
                    # A verified root rotation makes metadata version
                    # rollbacks safe (TUF fast-forward attack recovery)
                    self._allow_rollback = True

                except exceptions.DownloadHTTPError as exception:
                    if exception.status_code not in {403, 404}:
                        raise
                    # 404/403 means current root is newest available
                    break
        finally:
            # Make sure the non-versioned root.json links to current version
            self._update_root_symlink()

    def _load_timestamp(self) -> None:
        """Load local and remote timestamp metadata."""
        try:
            data = self._load_local_metadata(Timestamp.type)
            self._trusted_set.update_timestamp(data)
        except (OSError, exceptions.RepositoryError) as e:
            # Local timestamp does not exist or is invalid
            logger.debug("Local timestamp not valid as final: %s", e)

        # Load from remote (whether local load succeeded or not)
        remote_data = self._download_metadata(
            Timestamp.type, self.config.timestamp_max_length
        )
        try:
            self._trusted_set.update_timestamp(remote_data)
        except exceptions.EqualVersionNumberError:
            # If the new timestamp version is the same as current, discard the
            # new timestamp. This is normal and it shouldn't raise any error.
            # The locally stored timestamp stays part of the local metadata.
            return

        self._persist_metadata(Timestamp.type, remote_data)

    def _load_snapshot(self) -> None:
        """Load local (and if needed remote) snapshot metadata."""
        try:
            data = self._load_local_metadata(Snapshot.type)
            self._trusted_set.update_snapshot(data, trusted=True)
            logger.debug("Local snapshot is valid: not downloading new one")
        except (OSError, exceptions.RepositoryError) as e:
            # Local snapshot does not exist or is invalid: update from remote
            logger.debug("Local snapshot not valid as final: %s", e)

            snapshot_meta = self._trusted_set.timestamp.snapshot_meta
            length = snapshot_meta.length or self.config.snapshot_max_length
            version = None
            if self._trusted_set.root.consistent_snapshot:
                version = snapshot_meta.version

            data = self._download_metadata(Snapshot.type, length, version)
            self._trusted_set.update_snapshot(data)
            self._persist_metadata(Snapshot.type, data)

    def _load_targets(self, role: str, parent_role: str) -> Targets:
        """Load local (and if needed remote) metadata for ``role``."""

        # Avoid loading 'role' more than once for the same parent_role
        if self._trusted_set.contains(role, parent_role):
            return cast("Targets", self._trusted_set[role])

        try:
            data = self._load_local_metadata(role)
            delegated_targets = self._trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            logger.debug("Local %s is valid: not downloading new one", role)
            return delegated_targets
        except (OSError, exceptions.RepositoryError) as e:
            # Local 'role' does not exist or is invalid: update from remote
            logger.debug("Failed to load local %s: %s", role, e)

            snapshot = self._trusted_set.snapshot
            metainfo = snapshot.meta.get(f"{role}.json")
            if metainfo is None:
                raise exceptions.RepositoryError(
                    f"Role {role} was delegated but is not part of snapshot"
                ) from None

            length = metainfo.length or self.config.targets_max_length
            version = None
            if self._trusted_set.root.consistent_snapshot:
                version = metainfo.version

            data = self._download_metadata(role, length, version)
            delegated_targets = self._trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            with self._store.lock():
                # Another process may have committed while this Updater was
                # downloading: never replace newer verified metadata with older
                # metadata, so re-read the visible generation before merging.
                self._store.invalidate()
                self._local_generation = self._store.load()
                self._persist_metadata(role, data)
                self._commit_local_metadata()

            return delegated_targets

    def _preorder_depth_first_walk(
        self, target_filepath: str
    ) -> TargetFile | None:
        """
        Interrogates the tree of target delegations in order of appearance
        (which implicitly order trustworthiness), and returns the matching
        target found in the most trusted role.
        """

        # List of delegations to be interrogated. A (role, parent role) pair
        # is needed to load and verify the delegated targets metadata.
        delegations_to_visit = [(Targets.type, Root.type)]
        visited_role_names: set[str] = set()

        # Preorder depth-first traversal of the graph of target delegations.
        while (
            len(visited_role_names) <= self.config.max_delegations
            and len(delegations_to_visit) > 0
        ):
            # Pop the role name from the top of the stack.
            role_name, parent_role = delegations_to_visit.pop(-1)

            # Skip any visited current role to prevent cycles.
            if role_name in visited_role_names:
                logger.debug("Skipping visited current role %s", role_name)
                continue

            # The metadata for 'role_name' must be downloaded/updated before
            # its targets, delegations, and child roles can be inspected.
            targets = self._load_targets(role_name, parent_role)

            target = targets.targets.get(target_filepath)

            if target is not None:
                logger.debug("Found target in current role %s", role_name)
                return target

            # After preorder check, add current role to set of visited roles.
            visited_role_names.add(role_name)

            if targets.delegations is not None:
                child_roles_to_visit = []
                # NOTE: This may be a slow operation if there are many
                # delegated roles.
                for (
                    child_name,
                    terminating,
                ) in targets.delegations.get_roles_for_target(target_filepath):
                    logger.debug("Adding child role %s", child_name)
                    child_roles_to_visit.append((child_name, role_name))
                    if terminating:
                        logger.debug("Not backtracking to other roles")
                        delegations_to_visit = []
                        break
                # Push 'child_roles_to_visit' in reverse order of appearance
                # onto 'delegations_to_visit'.  Roles are popped from the end of
                # the list.
                child_roles_to_visit.reverse()
                delegations_to_visit.extend(child_roles_to_visit)

        if len(delegations_to_visit) > 0:
            logger.debug(
                "%d roles left to visit, but allowed at most %d delegations",
                len(delegations_to_visit),
                self.config.max_delegations,
            )

        # If this point is reached then target is not found, return None
        return None


def _ensure_trailing_slash(url: str) -> str:
    """Return url guaranteed to end in a slash."""
    return url if url.endswith("/") else f"{url}/"


def _metadata_version(data: bytes) -> int:
    """Return the signed version number of serialized metadata.

    The version is only used to avoid replacing newer verified metadata with
    older metadata; the bytes themselves have already been verified by
    ``TrustedMetadataSet`` before they reach this function.
    """
    try:
        document = json.loads(data)
        if "signed" in document:  # traditional metadata
            return int(document["signed"]["version"])
        # DSSE envelope: the payload is base64 encoded JSON
        payload = base64.b64decode(document["payload"])
        return int(json.loads(payload)["version"])
    except (ValueError, KeyError, TypeError):
        return 0
