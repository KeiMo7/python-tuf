# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""File-backed repository helpers for multi-process tests.

``RepositorySimulator`` lives in one process only. Tests that start several
client processes need a repository they can all read, so the parent process
dumps the simulator state with ``dump_repository()`` and the children read it
through ``FileFetcher``. URLs are of the form
``file:///<root>/metadata/<name>`` or ``file:///<root>/targets/<name>``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from urllib import parse, request

from tuf.api import exceptions
from tuf.ngclient.fetcher import FetcherInterface

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.repository_simulator import RepositorySimulator


class FileFetcher(FetcherInterface):
    """Serve repository metadata and targets from a local directory."""

    def _fetch(self, url: str) -> Iterator[bytes]:
        parsed = parse.urlparse(url)
        if parsed.scheme != "file":
            raise exceptions.DownloadError(f"Unsupported URL: {url}")
        path = request.url2pathname(parsed.path)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError as e:
            raise exceptions.DownloadHTTPError(f"Not found: {url}", 404) from e
        except OSError as e:
            raise exceptions.DownloadError(f"Failed to read {url}") from e
        yield data


def dump_repository(sim: RepositorySimulator, root_dir: str) -> None:
    """Write the current state of ``sim`` to ``root_dir``."""
    metadata_dir = os.path.join(root_dir, "metadata")
    targets_dir = os.path.join(root_dir, "targets")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    for version in range(1, len(sim.signed_roots) + 1):
        with open(
            os.path.join(metadata_dir, f"{version}.root.json"), "wb"
        ) as f:
            f.write(sim.fetch_metadata("root", version))

    consistent = sim.root.consistent_snapshot

    def write_role(role: str) -> None:
        data = sim.fetch_metadata(role)
        quoted = parse.quote(role, "")
        with open(os.path.join(metadata_dir, f"{quoted}.json"), "wb") as f:
            f.write(data)
        if consistent and role != "timestamp":
            number = json.loads(data)["signed"]["version"]
            with open(
                os.path.join(metadata_dir, f"{number}.{quoted}.json"), "wb"
            ) as f:
                f.write(data)

    for role in ("timestamp", "snapshot", "targets"):
        write_role(role)
    for role in sim.md_delegates:
        write_role(role)

    for target_path, repo_target in sim.target_files.items():
        destination = os.path.join(targets_dir, target_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(repo_target.data)
        if consistent:
            # consistent snapshots fetch hash-prefixed target paths
            dirname, sep, basename = target_path.rpartition("/")
            for digest in repo_target.target_file.hashes.values():
                prefixed = os.path.join(
                    targets_dir, f"{dirname}{sep}{digest}.{basename}"
                )
                with open(prefixed, "wb") as f:
                    f.write(repo_target.data)
