# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Child process helper for crash safety and multi-process tests.

Started by ``tests/test_local_store.py``; kills itself at a well defined point
of the refresh/commit protocol, simulating ``kill -9`` or a power loss at that
exact point. The parent process then inspects the metadata directory.

Usage: ``_crash_helper.py <hook>`` where hook is one of

  * ``staging``: while the new generation is still being written
  * ``renamed``: after the generation directory was renamed into place but
    before the pointer was replaced (the commit point)
  * ``committed``: right after the pointer was atomically replaced
  * ``target``: while a target file is being written
  * ``lock``: takes the metadata lock and waits forever (the parent kills the
    process to prove that the OS releases the lock)
  * ``refresh``: a normal refresh that prints the versions it sees as JSON

Environment:
  TUF_CRASH_METADATA_DIR, TUF_CRASH_TARGETS_DIR: client directories
  TUF_CRASH_REPO_DIR: directory containing the dumped repository
  TUF_CRASH_REPO_VERSION: repository (targets) version to expect
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Running with ``-I`` ignores PYTHONPATH: make the checkout importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.file_repository import FileFetcher
from tuf.ngclient import Updater, UpdaterConfig
from tuf.ngclient._internal import local_store


def _updater(
    metadata_dir: str,
    targets_dir: str,
    repo_dir: str,
    *,
    lock_timeout: float = 30.0,
) -> Updater:
    root_path = os.path.join(metadata_dir, "root.json")
    if os.path.exists(root_path):
        with open(root_path, "rb") as f:
            bootstrap = f.read()
    else:
        with open(os.path.join(repo_dir, "metadata", "1.root.json"), "rb") as f:
            bootstrap = f.read()
    return Updater(
        metadata_dir,
        f"file://{repo_dir}/metadata/",
        targets_dir,
        f"file://{repo_dir}/targets/",
        FileFetcher(),
        config=UpdaterConfig(lock_timeout=lock_timeout),
        bootstrap=bootstrap,
    )


def _install_hook(updater: Updater, hook: str) -> None:
    """Install a crash point into the store used by ``updater``."""
    store = updater._store
    if hook == "staging":

        def commit(*_args: object, **_kwargs: object) -> None:
            # leave a half-written staging directory behind, like a kill -9
            staging = os.path.join(store._generations_dir, ".staging-crash")
            os.makedirs(staging, exist_ok=True)
            with open(os.path.join(staging, "partial.json"), "wb") as f:
                f.write(b"{")
            os._exit(17)

        store.commit = commit  # type: ignore[method-assign]
    elif hook == "renamed":
        original_pointer = store._write_pointer

        def write_pointer_renamed(
            _generation: local_store.Generation,
        ) -> None:
            # generation is renamed into place but never made visible
            os._exit(18)

        store._write_pointer = write_pointer_renamed  # type: ignore[method-assign]
    elif hook == "committed":
        original_pointer = store._write_pointer

        def write_pointer_committed(
            generation: local_store.Generation,
        ) -> None:
            original_pointer(generation)
            os._exit(19)

        store._write_pointer = write_pointer_committed  # type: ignore[method-assign]
    elif hook == "target":
        targets_dir = updater.target_dir or ""
        original_replace = os.replace

        def replace(source: str, destination: str) -> None:
            if os.path.dirname(destination) == targets_dir:
                # leave the temporary file behind, like a kill -9
                os._exit(20)
            original_replace(source, destination)

        os.replace = replace  # type: ignore[assignment]


def main() -> None:
    hook = sys.argv[1]
    metadata_dir = os.environ["TUF_CRASH_METADATA_DIR"]
    targets_dir = os.environ["TUF_CRASH_TARGETS_DIR"]
    repo_dir = os.environ["TUF_CRASH_REPO_DIR"]

    if os.environ.get("TUF_CRASH_DEBUG"):
        logging.basicConfig(
            filename=os.environ["TUF_CRASH_DEBUG"],
            level=logging.INFO,
            format="%(name)s %(levelname)s %(message)s",
        )

    if hook == "lock":
        store = local_store.MetadataStore(metadata_dir)
        with store.lock():
            print("locked", flush=True)
            # never return: the parent kills this process
            while True:
                pass  # pragma: no cover

    lock_timeout = float(os.environ.get("TUF_CRASH_LOCK_TIMEOUT", "30"))
    updater = _updater(
        metadata_dir, targets_dir, repo_dir, lock_timeout=lock_timeout
    )

    if hook == "refresh":
        updater.refresh()
        generation = updater._store.verify()
        print(
            json.dumps(
                {
                    "versions": updater.local_metadata_versions(),
                    "generation": None
                    if generation is None
                    else generation.number,
                }
            ),
            flush=True,
        )
        return

    _install_hook(updater, hook)
    if hook == "target":
        updater.refresh()
        info = updater.get_targetinfo("target")
        assert info is not None
        updater.download_target(info)
        return

    updater.refresh()


if __name__ == "__main__":
    main()
