# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Crash-safe, multi-process safe storage of verified local metadata.

``Updater`` keeps the metadata it has verified on disk so that a later run can
reuse it.  A plain set of files named after their roles cannot be updated
atomically: a crash or a concurrent writer can leave a mixture of old and new
files, and a reader then has no way to tell which combination is the one that
was actually trusted.

``MetadataStore`` therefore stores each verified metadata set as an immutable
*generation* directory.  A single, tiny pointer file is the commit point; it is
replaced atomically, so a reader that resolves the pointer once observes exactly
one complete generation for its whole lifetime.  Generations are never modified
after they have been made visible.

Layout of a metadata directory::

    <metadata_dir>/
        current                     # atomically replaced pointer
        lock                        # cross-process mutex + owner record
        root.json                   # trust anchor (-> root_history/*.root.json)
        root_history/1.root.json    # append-only root chain
        generations/
            gen-0000000001/
                manifest.json
                root.json
                timestamp.json
                snapshot.json
                targets.json
                <delegated role>.json

The root chain is also kept as an append-only, independently verified history
(``root_history/``) so that a newer root can be verified even if a metadata
commit failed, and so the ``root.json`` anchor can point at any verified root.

Commit protocol (performed while holding the lock)::

    1. write files and manifest into generations/.staging-<uuid>/
    2. fsync files, manifest and directory
    3. rename staging directory to generations/gen-<n>   (generation exists)
    4. atomically replace "current" with a pointer to gen-<n>  (commit point)
    5. fsync metadata directory

If the process dies before step 4 the previous pointer is still in place and the
previous generation remains the visible one.  A generation that was renamed but
never pointed at is complete by construction, so recovery adopts the newest
complete generation and logs that it did so.  A generation whose manifest or
files are damaged is skipped and recovery falls back to the newest complete
older generation; if there is none an ``IncompleteMetadataGenerationError`` is
raised instead of trusting a partial set.

Exclusive access is provided by the operating system (``flock`` on POSIX,
``msvcrt.locking`` on Windows).  Because the kernel releases these locks when
the holder dies, a killed process can never leave a lock behind.  The lock file
also contains a human readable owner record that is refreshed while the lock is
held, for diagnostics and for diagnosing a lock holder that is alive but stuck.
Waiting for the lock is bounded: on timeout ``LockTimeoutError`` is raised and
the caller must fail.  The store never writes metadata without the lock.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib import parse

from tuf.api import exceptions
from tuf.api.metadata import Root

if os.name == "nt":  # pragma: no cover - windows only
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger(__name__)

_POINTER_NAME = "current"
_LOCK_NAME = "lock"
_GENERATIONS_DIR = "generations"
_MANIFEST_NAME = "manifest.json"
_FORMAT_VERSION = 1
_STAGING_PREFIX = ".staging-"
_GENERATION_PREFIX = "gen-"
_GENERATION_DIGITS = 10


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(path: str) -> None:
    """Make a directory entry durable (best effort: not possible on Windows)."""
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(path: str, data: bytes) -> None:
    """Write data to path, fsyncing file and directory before returning."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(os.path.dirname(path))


@dataclass(frozen=True)
class Generation:
    """An immutable, complete set of locally verified metadata.

    ``roles`` maps each stored role name to the metadata version and is the
    independent record of what was trusted; ``manifest_digest`` identifies the
    exact manifest file that was committed.
    """

    number: int
    path: str
    roles: Mapping[str, dict[str, object]]
    manifest_digest: str

    @property
    def versions(self) -> dict[str, int]:
        """Return role name -> metadata version for this generation."""
        versions: dict[str, int] = {}
        for role, info in self.roles.items():
            version = info.get("version")
            if isinstance(version, int):
                versions[role] = version
        return versions

    def read(self, role: str) -> bytes:
        """Return the stored bytes for ``role``.

        Raises:
            FileNotFoundError: ``role`` is not part of this generation.
        """
        info = self.roles.get(role)
        if info is None:
            raise FileNotFoundError(
                f"Role '{role}' is not in local generation {self.number}"
            )
        filename = str(info["filename"])
        with open(os.path.join(self.path, filename), "rb") as f:
            return f.read()


class MetadataStore:
    """Exclusive, generation based storage for verified local metadata."""

    def __init__(
        self,
        metadata_dir: str,
        lock_timeout: float = 30.0,
        lease_timeout: float | None = 300.0,
        generations_to_keep: int = 5,
    ) -> None:
        self._dir = metadata_dir
        self._generations_dir = os.path.join(metadata_dir, _GENERATIONS_DIR)
        self._pointer_path = os.path.join(metadata_dir, _POINTER_NAME)
        self._lock_path = os.path.join(metadata_dir, _LOCK_NAME)
        self._lock_timeout = lock_timeout
        self._lease_timeout = lease_timeout
        self._generations_to_keep = max(1, generations_to_keep)

        # resolved generation cache: keeps one Updater on one generation until
        # it commits a new one
        self._cached: Generation | None = None
        self._cached_resolved = False

        # lock state (re-entrant within one store instance)
        self._lock_fd: int | None = None
        self._lock_depth = 0
        self._lock_owner_id: str | None = None

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Take exclusive access to the metadata directory.

        Waits at most ``lock_timeout`` seconds.  Raises ``LockTimeoutError`` if
        the lock cannot be acquired in time: callers must not write without it.

        The lock is re-entrant for a single store instance.
        """
        if self._lock_fd is not None:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return

        os.makedirs(self._dir, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._acquire_os_lock(fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise

        self._lock_fd = fd
        self._lock_depth = 1
        self._lock_owner_id = uuid.uuid4().hex
        try:
            self._write_owner_record(heartbeat=True)
            yield
        finally:
            self._lock_depth = 0
            with contextlib.suppress(OSError):
                self._write_owner_record(heartbeat=False)
            self._lock_fd = None
            self._lock_owner_id = None
            self._release_os_lock(fd)

    def _acquire_os_lock(self, fd: int) -> None:
        deadline = time.monotonic() + self._lock_timeout
        delay = 0.001
        while True:
            if self._try_lock(fd):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise exceptions.LockTimeoutError(
                    f"Could not acquire metadata lock '{self._lock_path}'"
                    f" within {self._lock_timeout} seconds:"
                    f" {self._holder_info()}"
                )
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.1)

    @staticmethod
    def _try_lock(fd: int) -> bool:
        try:
            if os.name == "nt":  # pragma: no cover - windows only
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    @staticmethod
    def _release_os_lock(fd: int) -> None:
        with contextlib.suppress(OSError):
            if os.name == "nt":  # pragma: no cover - windows only
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    def _write_owner_record(self, heartbeat: bool) -> None:
        """Write a diagnostic owner record into the lock file.

        The OS lock is the actual mutex; this record only makes it possible to
        explain who holds the lock (and for how long) in logs and error
        messages.  A record older than ``lease_timeout`` identifies a holder
        that is alive but not making progress.
        """
        if self._lock_fd is None:
            return
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "owner": self._lock_owner_id,
            "heartbeat": time.time(),
            "released": not heartbeat,
        }
        data = json.dumps(record, sort_keys=True).encode("utf-8")
        with contextlib.suppress(OSError):
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            os.ftruncate(self._lock_fd, 0)
            os.write(self._lock_fd, data)
            os.fsync(self._lock_fd)

    def _read_owner_record(self) -> dict[str, object] | None:
        try:
            with open(self._lock_path, "rb") as f:
                return json.loads(f.read() or b"null")
        except (OSError, ValueError):
            return None

    def _holder_info(self) -> str:
        record = self._read_owner_record()
        if not record:
            return "lock holder is unknown"
        heartbeat = record.get("heartbeat")
        age = ""
        if isinstance(heartbeat, int | float):
            age = f", last heartbeat {time.time() - float(heartbeat):.1f}s ago"
        stale = ""
        if (
            self._lease_timeout is not None
            and isinstance(heartbeat, int | float)
            and time.time() - float(heartbeat) > self._lease_timeout
        ):
            stale = " (lease appears expired: holder may be stuck)"
        return (
            f"held by pid {record.get('pid')} on {record.get('host')}"
            f"{age}{stale}"
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load(self) -> Generation | None:
        """Return the current complete generation, or ``None`` if none exists.

        The pointer is resolved and cached for the lifetime of the store, so a
        caller reading multiple roles can never mix generations.  If a newer
        complete generation exists (a writer died between renaming its
        generation and updating the pointer) it is adopted; damaged generations
        are skipped and the newest complete older generation is used.

        Raises:
            IncompleteMetadataGenerationError: A generation was visible, but no
                complete generation could be found.
        """
        if self._cached_resolved:
            return self._cached

        pointer = self._read_pointer()
        generation = self._generation_from_pointer(pointer)

        newest = self._newest_generation_number()
        if generation is not None and (
            newest is None or newest <= generation.number
        ):
            self._remember(generation)
            return generation

        # There may be a newer complete generation: adopt it under the lock so
        # that two processes cannot race while doing recovery.
        try:
            with self.lock():
                generation = self._recover(pointer, generation)
        except exceptions.LockTimeoutError:
            if generation is not None:
                # A valid generation is visible: never block a reader on
                # recovery, and never fall back to a partial set.
                logger.warning(
                    "Could not acquire metadata lock for recovery, using "
                    "visible local generation %d",
                    generation.number,
                )
                self._remember(generation)
                return generation
            raise

        self._remember(generation)
        return generation

    def read(self, role: str) -> bytes:
        """Read ``role`` from the current generation.

        If the file vanishes between resolving the pointer and opening it, a
        concurrent writer may have pruned the generation: the pointer is
        resolved again and the read retried once. This keeps readers lock-free.

        Raises:
            FileNotFoundError: There is no current generation or it does not
                contain ``role``.
        """
        for attempt in range(2):
            generation = self.load()
            if generation is None:
                raise FileNotFoundError(
                    f"No local metadata generation in '{self._dir}'"
                )
            try:
                return generation.read(role)
            except FileNotFoundError:
                if attempt:
                    raise
                self.invalidate()
        raise AssertionError("unreachable")

    def invalidate(self) -> None:
        """Forget the cached generation so the next read resolves again."""
        self._cached = None
        self._cached_resolved = False

    def recover(self) -> Generation | None:
        """Recover and adopt the newest complete generation. Lock must be held.

        A generation that was renamed into place but never made visible by a
        process that died is adopted; a damaged visible generation falls back
        to the newest complete older generation (and the rollback is logged) or
        raises ``IncompleteMetadataGenerationError`` when there is none.
        """
        if self._lock_fd is None:
            raise RuntimeError("Metadata store lock is not held")
        pointer = self._read_pointer()
        generation = self._generation_from_pointer(pointer)
        adopted = self._recover(pointer, generation)
        self._remember(adopted)
        return adopted

    def _remember(self, generation: Generation | None) -> None:
        self._cached = generation
        self._cached_resolved = True

    def _read_pointer(self) -> dict[str, object] | None:
        try:
            with open(self._pointer_path, "rb") as f:
                pointer = json.loads(f.read())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.error(
                "Local metadata pointer '%s' is unreadable: %s",
                self._pointer_path,
                e,
            )
            return {"corrupt": True}
        if not isinstance(pointer, dict):
            logger.error("Local metadata pointer is not an object")
            return {"corrupt": True}
        return pointer

    def _generation_from_pointer(
        self, pointer: dict[str, object] | None
    ) -> Generation | None:
        """Load the generation a pointer refers to, structurally validated."""
        if pointer is None or pointer.get("corrupt"):
            return None
        number = pointer.get("generation")
        digest = pointer.get("manifest")
        if not isinstance(number, int) or not isinstance(digest, str):
            logger.error("Local metadata pointer is malformed: %s", pointer)
            return None
        try:
            generation = self._load_generation(number, verify_hashes=False)
        except (OSError, ValueError) as e:
            logger.error(
                "Local metadata generation %d is incomplete: %s", number, e
            )
            return None
        if generation.manifest_digest != digest:
            logger.error(
                "Local metadata generation %d does not match its pointer",
                number,
            )
            return None
        return generation

    def _load_generation(self, number: int, verify_hashes: bool) -> Generation:
        path = self._generation_path(number)
        manifest_path = os.path.join(path, _MANIFEST_NAME)
        with open(manifest_path, "rb") as f:
            raw = f.read()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        if manifest.get("format") != _FORMAT_VERSION:
            raise ValueError(
                f"unknown manifest format {manifest.get('format')}"
            )
        if manifest.get("generation") != number:
            raise ValueError("manifest generation does not match directory")
        roles = manifest.get("files")
        if not isinstance(roles, dict):
            raise ValueError("manifest files missing")

        for role, info in roles.items():
            _verify_manifest_entry(path, role, info, verify_hashes)

        return Generation(number, path, roles, _sha256(raw))

    def _generation_path(self, number: int) -> str:
        return os.path.join(
            self._generations_dir,
            f"{_GENERATION_PREFIX}{number:0{_GENERATION_DIGITS}d}",
        )

    def _generation_numbers(self) -> list[int]:
        """Return generation numbers present on disk (complete or not)."""
        try:
            names = os.listdir(self._generations_dir)
        except FileNotFoundError:
            return []
        numbers = []
        for name in names:
            if not name.startswith(_GENERATION_PREFIX):
                continue
            try:
                numbers.append(int(name[len(_GENERATION_PREFIX) :]))
            except ValueError:
                continue
        return sorted(numbers)

    def _newest_generation_number(self) -> int | None:
        numbers = self._generation_numbers()
        return numbers[-1] if numbers else None

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover(
        self, pointer: dict[str, object] | None, visible: Generation | None
    ) -> Generation | None:
        """Find and adopt the newest complete generation. Lock must be held."""
        if self._lock_fd is None:
            raise RuntimeError("Metadata store lock is not held")
        self._remove_staging_dirs()

        pointer_number = _pointer_generation(pointer)
        pointer_digest = (
            pointer.get("manifest") if isinstance(pointer, dict) else None
        )
        adopted: Generation | None = None
        for number in reversed(self._generation_numbers()):
            try:
                generation = self._load_generation(number, verify_hashes=True)
            except (OSError, ValueError) as e:
                logger.error(
                    "Discarding incomplete local metadata generation %d: %s",
                    number,
                    e,
                )
                continue
            if (
                number == pointer_number
                and isinstance(pointer_digest, str)
                and generation.manifest_digest != pointer_digest
            ):
                # The generation the pointer refers to was modified after it
                # was committed: do not adopt it.
                logger.error(
                    "Discarding local metadata generation %d: manifest does "
                    "not match the pointer",
                    number,
                )
                continue
            adopted = generation
            break

        if adopted is None:
            if pointer is not None:
                raise exceptions.IncompleteMetadataGenerationError(
                    "Local metadata is damaged and no complete generation "
                    f"is available in '{self._dir}'"
                )
            return None

        current_number = pointer_number
        if visible is None or adopted.number != visible.number:
            if current_number is not None and adopted.number < current_number:
                logger.error(
                    "Local metadata generation %d is incomplete: rolling back "
                    "to complete generation %d",
                    current_number,
                    adopted.number,
                )
            else:
                logger.warning(
                    "Adopting complete local metadata generation %d found on "
                    "disk",
                    adopted.number,
                )
            self._write_pointer(adopted)

        self._prune(adopted)
        return adopted

    def _remove_staging_dirs(self) -> None:
        try:
            names = os.listdir(self._generations_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if not name.startswith(_STAGING_PREFIX):
                continue
            path = os.path.join(self._generations_dir, name)
            logger.warning("Removing stale metadata staging directory %s", path)
            shutil.rmtree(path, ignore_errors=True)

        # temp files from a writer that died just before its atomic rename
        try:
            root_names = os.listdir(self._dir)
        except FileNotFoundError:
            return
        for name in root_names:
            if name.startswith((".current-", ".root-")):
                path = os.path.join(self._dir, name)
                logger.warning("Removing stale temporary file %s", path)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

    def _prune(self, current: Generation) -> None:
        """Remove old generations, keeping the current and recent history."""
        numbers = self._generation_numbers()
        keep = {current.number}
        older = [n for n in numbers if n < current.number]
        keep.update(older[-self._generations_to_keep :])
        for number in numbers:
            if number in keep:
                continue
            path = self._generation_path(number)
            logger.debug("Removing old metadata generation %d", number)
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def commit(
        self,
        base: Generation | None,
        files: Mapping[str, bytes],
        versions: Mapping[str, int],
        *,
        root: tuple[int, bytes] | None = None,
        allow_rollback: bool = False,
    ) -> Generation:
        """Store ``files`` as a new complete generation and make it visible.

        ``files`` maps role name to already verified metadata bytes.  Roles not
        in ``files`` are carried over from ``base`` (hard linked when possible:
        generation files are immutable) so that a generation is always a
        complete set.  Every file in a generation was verified by the TUF
        client workflow; on load the whole set is verified again from the
        trusted root, so a reader never has to repair anything.

        If ``root`` is given, the root metadata is part of the same atomic
        commit: readers never see a new root together with an old metadata set
        (or the reverse).

        ``allow_rollback`` permits committing metadata with a lower version
        than ``base``. This is only correct after a root key rotation, which
        makes a rollback safe (see the fast-forward recovery cases in the TUF
        specification). Without it, a writer that lost a race can never roll a
        role back.

        Lock must be held.

        Raises:
            RuntimeError: lock is not held.
            OSError: generation could not be written.
        """
        if self._lock_fd is None:
            raise RuntimeError("Metadata store lock is not held")

        if base is not None and not allow_rollback:
            files, versions = self._drop_stale(base, files, versions)
            if not files and root is None:
                logger.info(
                    "Local metadata generation %d is already at least as new "
                    "as the verified metadata",
                    base.number,
                )
                return base

        number = self._next_generation_number()
        manifest_files, manifest_bytes = self._stage(
            number, base, files, versions, root
        )

        generation = Generation(
            number,
            self._generation_path(number),
            manifest_files,
            _sha256(manifest_bytes),
        )
        self._write_pointer(generation)
        logger.info(
            "Committed local metadata generation %d (%s): %s",
            generation.number,
            generation.manifest_digest,
            ", ".join(
                f"{role} v{info['version']}"
                for role, info in sorted(manifest_files.items())
            ),
        )
        self._prune(generation)
        self._remember(generation)
        return generation

    @staticmethod
    def _drop_stale(
        base: Generation,
        files: Mapping[str, bytes],
        versions: Mapping[str, int],
    ) -> tuple[dict[str, bytes], dict[str, int]]:
        """Remove files whose version is older than the committed generation."""
        base_versions = base.versions
        stale = {
            role: version
            for role, version in versions.items()
            if version < base_versions.get(role, -1)
        }
        if not stale:
            return dict(files), dict(versions)
        logger.warning(
            "Not replacing newer local metadata in generation %d: %s",
            base.number,
            ", ".join(
                f"{role} v{version} < v{base_versions[role]}"
                for role, version in sorted(stale.items())
            ),
        )
        return (
            {role: data for role, data in files.items() if role not in stale},
            {
                role: version
                for role, version in versions.items()
                if role not in stale
            },
        )

    def _stage(
        self,
        number: int,
        base: Generation | None,
        files: Mapping[str, bytes],
        versions: Mapping[str, int],
        root: tuple[int, bytes] | None,
    ) -> tuple[dict[str, dict[str, object]], bytes]:
        """Write a complete generation to a staging dir, then rename it.

        Returns the manifest entries and the exact manifest bytes.
        """
        staging = os.path.join(
            self._generations_dir, f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
        )
        os.makedirs(staging)
        try:
            manifest_files: dict[str, dict[str, object]] = {}
            staged_roles = set(files)

            # carry over roles that are not part of this commit
            if base is not None:
                for role, info in base.roles.items():
                    if role in staged_roles:
                        continue
                    if root is not None and role == Root.type:
                        # replaced by the root in this commit
                        continue
                    filename = str(info["filename"])
                    _link_or_copy(
                        os.path.join(base.path, filename),
                        os.path.join(staging, filename),
                    )
                    manifest_files[role] = dict(info)

            if root is not None:
                root_version, root_bytes = root
                filename = _role_filename(Root.type)
                _write_all(os.path.join(staging, filename), root_bytes)
                manifest_files[Root.type] = {
                    "filename": filename,
                    "length": len(root_bytes),
                    "sha256": _sha256(root_bytes),
                    "version": int(root_version),
                }

            for role, data in files.items():
                filename = _role_filename(role)
                _write_all(os.path.join(staging, filename), data)
                manifest_files[role] = {
                    "filename": filename,
                    "length": len(data),
                    "sha256": _sha256(data),
                    "version": int(versions[role]),
                }

            manifest = {
                "format": _FORMAT_VERSION,
                "generation": number,
                "files": manifest_files,
            }
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, indent=1
            ).encode("utf-8")
            _write_all(os.path.join(staging, _MANIFEST_NAME), manifest_bytes)
            _fsync_dir(staging)
            os.rename(staging, self._generation_path(number))
            _fsync_dir(self._generations_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest_files, manifest_bytes

    def _next_generation_number(self) -> int:
        newest = self._newest_generation_number()
        return 1 if newest is None else newest + 1

    def _write_pointer(self, generation: Generation) -> None:
        """Atomically replace the pointer: this is the commit point."""
        pointer = {
            "format": _FORMAT_VERSION,
            "generation": generation.number,
            "manifest": generation.manifest_digest,
        }
        data = json.dumps(pointer, sort_keys=True).encode("utf-8")
        fd, temp_path = tempfile.mkstemp(
            dir=self._dir, prefix=".current-", suffix=".tmp"
        )
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            os.replace(temp_path, self._pointer_path)
            _fsync_dir(self._dir)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_path)
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def verify(self) -> Generation | None:
        """Verify the visible generation (hashes included).

        Resolves the visible generation exactly like :meth:`load` (including
        adopting a newer complete generation left behind by a killed writer)
        and additionally verifies every file hash. A damaged generation is
        rolled back to the newest complete older generation, and the rollback
        is reported. Intended for diagnostics and tests: normal reads only do
        structural checks so that a metadata read does not cost a second hash
        pass.

        Raises:
            IncompleteMetadataGenerationError: no complete generation exists.
            LockTimeoutError: recovery was needed but the lock was busy.
        """
        generation = self.load()
        if generation is None:
            return None
        try:
            verified = self._load_generation(
                generation.number, verify_hashes=True
            )
        except (OSError, ValueError) as e:
            logger.error(
                "Local metadata generation %d is incomplete: %s",
                generation.number,
                e,
            )
            with self.lock():
                recovered = self._recover(self._read_pointer(), visible=None)
            self._remember(recovered)
            if recovered is None:
                raise exceptions.IncompleteMetadataGenerationError(
                    f"No complete local metadata generation in '{self._dir}'"
                ) from e
            return recovered

        if verified.manifest_digest != generation.manifest_digest:
            logger.error(
                "Local metadata generation %d does not match its manifest",
                generation.number,
            )
            with self.lock():
                recovered = self._recover(self._read_pointer(), visible=None)
            self._remember(recovered)
            if recovered is None:
                raise exceptions.IncompleteMetadataGenerationError(
                    "Local metadata generation does not match its pointer"
                )
            return recovered
        return verified


def _pointer_generation(pointer: Mapping[str, object] | None) -> int | None:
    """Return the generation number a pointer refers to, if valid."""
    if pointer is None:
        return None
    number = pointer.get("generation")
    return number if isinstance(number, int) else None


def _verify_manifest_entry(
    path: str, role: str, info: object, verify_hashes: bool
) -> None:
    """Validate one manifest entry and the file it refers to."""
    if not isinstance(info, dict):
        raise ValueError(f"manifest entry for {role} is invalid")
    filename = info.get("filename")
    length = info.get("length")
    digest = info.get("sha256")
    if (
        not isinstance(filename, str)
        or not isinstance(length, int)
        or not isinstance(digest, str)
    ):
        raise ValueError(f"manifest entry for {role} is invalid")
    file_path = os.path.join(path, filename)
    stat = os.stat(file_path)
    if stat.st_size != length:
        raise ValueError(
            f"'{filename}' has size {stat.st_size}, expected {length}"
        )
    if verify_hashes:
        with open(file_path, "rb") as f:
            if _sha256(f.read()) != digest:
                raise ValueError(f"'{filename}' does not match its hash")


def _role_filename(role: str) -> str:
    return f"{parse.quote(role, '')}.json"


def _link_or_copy(source: str, destination: str) -> None:
    """Hard link an immutable generation file, copying if linking fails."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
