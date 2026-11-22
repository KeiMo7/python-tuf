Concurrent use and crash safety
================================

An application may run more than one ``Updater`` against the same
``metadata_dir`` and ``target_dir``: separate processes, separate threads, or
one process on several build workers sharing a cache directory. The ngclient
is designed so that this is safe without any cooperation between the callers.

Why plain files are not enough
------------------------------

Storing one file per role (``timestamp.json``, ``snapshot.json``, ...) and
overwriting them in place has three failure modes:

* a refresh writes several files: a crash between writes leaves a mixture of
  old and new metadata
* two processes refreshing at once can interleave their writes
* a reader cannot tell which versions belong together, so it either trusts a
  mixed set or has to repair the directory itself

The local metadata store
------------------------

Every successfully verified metadata set is written as an immutable
*generation* directory, and a single small pointer file is the commit point::

   <metadata_dir>/
       current                     # atomically replaced pointer
       lock                        # cross-process mutex
       root.json                   # trust anchor
       root_history/1.root.json    # append-only root chain
       generations/
           gen-0000000001/
               manifest.json
               root.json
               timestamp.json
               snapshot.json
               targets.json
               <delegated role>.json

A refresh does all network access and all signature, hash, expiry and rollback
verification *before* it takes the lock. It then writes the complete set into a
staging directory, fsyncs it, renames it to ``gen-<n>`` and finally replaces
``current`` with an atomic rename. Until that last rename nothing is visible:
a process that is killed at any point leaves the previous generation as the
visible one.

A reader resolves ``current`` once and reads every role from that one
generation, so it always sees a self-consistent set. It never has to repair or
mix anything.

Loading and recovery
--------------------

When a generation is found that was renamed into place but never made visible
(the writer died between the rename and the pointer update), the next process
adopts it: it was already complete and verified. If the visible generation is
damaged, the store falls back to the newest complete older generation, logs the
rollback and persists it. If no complete generation is left,
``IncompleteMetadataGenerationError`` is raised rather than trusting a partial
set.

The ``current`` pointer and the ``root.json`` anchor are also the independent
record of *which* metadata version this client trusts: an Updater logs it after
every refresh (``logging`` topic ``tuf.ngclient.updater``) and exposes it as
``Updater.local_metadata_versions()``. This makes a later investigation
possible without guessing from file contents.

Mutual exclusion and leases
---------------------------

The lock is an operating system lock (``flock`` on POSIX, ``msvcrt.locking``
on Windows). Because the kernel releases it when the holder dies, a killed
process can never leave the directory locked. A small owner record (pid, host,
last heartbeat) inside the lock file is used for diagnostics.

Waiting is bounded by ``UpdaterConfig.lock_timeout`` (default 30 seconds). When
it expires ``LockTimeoutError`` is raised: the updater fails explicitly instead
of writing without the lock. ``UpdaterConfig.lease_timeout`` only identifies a
lock holder that is alive but no longer making progress (for diagnostics).

Target files
------------

``download_target()`` writes the target into a temporary file next to the
destination, verifies length and hashes, fsyncs it and then renames it into
place. A killed download therefore never leaves a partial file at the visible
path, and ``find_cached_target()`` re-verifies any file it finds. A target that
fails verification is never published.

Tuning
------

``UpdaterConfig`` has three relevant options:

* ``lock_timeout``: how long to wait for exclusive access (default 30s)
* ``lease_timeout``: age after which a lock owner record is reported as stuck
  (default 300s)
* ``generations_to_keep``: how many older generations are kept for recovery and
  rollback (default 5)

For a single-process workload the additional work is one small pointer read at
startup and one generation commit per refresh.

Reproducing the scenarios
-------------------------

Step-by-step instructions for reproducing a concurrent conflict and a forced
termination are in `REPRODUCING_CONCURRENCY.md
<https://github.com/theupdateframework/python-tuf/blob/develop/docs/REPRODUCING_CONCURRENCY.md>`_.
The corresponding tests are in ``tests/test_local_store.py``.
