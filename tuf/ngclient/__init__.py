# Copyright 2021-2025, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""TUF client public API."""

from tuf.api.metadata import TargetFile
from tuf.ngclient.audit import (
    AuditCorruption,
    AuditLog,
    AuditLogError,
    AuditLogFormatError,
    AuditLogVersionError,
    AuditLogWriteError,
    UpdateAuditor,
)
from tuf.ngclient.config import UpdaterConfig
from tuf.ngclient.fetcher import FetcherInterface
from tuf.ngclient.updater import Updater
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

__all__ = [  # noqa: PLE0604
    FetcherInterface.__name__,
    Urllib3Fetcher.__name__,
    TargetFile.__name__,
    Updater.__name__,
    UpdaterConfig.__name__,
    AuditLog.__name__,
    UpdateAuditor.__name__,
    AuditCorruption.__name__,
    AuditLogError.__name__,
    AuditLogFormatError.__name__,
    AuditLogVersionError.__name__,
    AuditLogWriteError.__name__,
]
