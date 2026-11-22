# Reproducing concurrency and crash scenarios

These steps are self-contained: they use only this checkout and its test
helpers, so another person can repeat them and reach the same conclusions.

Run all commands from the repository root with the test dependencies installed:

```bash
python3 -m pip install -r requirements/test.txt
```

## 1. Concurrent refreshes (four processes, one metadata directory)

The multi-process tests dump a repository to a local directory, start four
independent Python processes that each run `Updater.refresh()`, and assert that

* every process reports the same versions,
* the visible generation is one complete generation,
* reader threads running during the commits never see a timestamp/snapshot/
  targets combination that did not come from one generation.

```bash
python3 -m unittest -v \
  tests.test_local_store.TestMultiProcess.test_concurrent_first_refresh
python3 -m unittest -v \
  tests.test_local_store.TestMultiProcess.test_concurrent_refresh_with_concurrent_readers
```

To do it by hand, create a repository dump and four concurrent refreshers:

```bash
export TMP=$(mktemp -d)
python3 - "$TMP" <<'PY'
import os, sys
from tests.repository_simulator import RepositorySimulator
from tests.file_repository import dump_repository

tmp = sys.argv[1]
for name in ("metadata", "targets"):
    os.mkdir(os.path.join(tmp, name))
dump_repository(RepositorySimulator(), os.path.join(tmp, "repo"))
PY

for i in 1 2 3 4; do
  TUF_CRASH_METADATA_DIR=$TMP/metadata \
  TUF_CRASH_TARGETS_DIR=$TMP/targets \
  TUF_CRASH_REPO_DIR=$TMP/repo \
  PYTHONPATH=. python3 tests/_crash_helper.py refresh &
done
wait
```

Each process prints the versions it trusts (JSON). They all print the same
values. Inspect what is on disk:

```bash
cat $TMP/metadata/current
python3 - "$TMP" <<'PY'
import json, os, sys
tmp = sys.argv[1]
pointer = json.load(open(os.path.join(tmp, "metadata", "current")))
gen = os.path.join(
    tmp, "metadata", "generations", f"gen-{pointer['generation']:010d}"
)
print(pointer)
print(sorted(os.listdir(gen)))
PY
```

Expected: one pointer, one complete generation (manifest + all roles), and
identical version JSON from every helper. Re-running the helpers does not
change the result.

## 2. Forced termination at each write point

`tests/_crash_helper.py` kills itself (`os._exit`) at a specific point of the
commit protocol:

| hook        | exit code | killed while                                   |
|-------------|-----------|------------------------------------------------|
| `staging`   | 17        | the new generation is being written            |
| `renamed`   | 18        | the generation is renamed, pointer not updated |
| `committed` | 19        | the pointer was just replaced                  |
| `target`    | 20        | a target file is being written                 |

The test asserts, for every hook, that after the crash a fresh process can read
the directory, that the visible metadata is a complete generation, and that
versions never go backwards:

```bash
python3 -m unittest -v tests.test_local_store.TestCrashRecovery
```

Manual run of the interesting case (`renamed`: the process dies after the new
generation directory exists but before it becomes visible). Note that the
repository simulator must stay in one process, so the whole scenario runs in a
single Python script that starts the crashing child:

```bash
python3 - <<'PY'
import json, os, subprocess, sys, tempfile

from tests.file_repository import dump_repository
from tests.repository_simulator import RepositorySimulator

tmp = tempfile.mkdtemp()
for name in ("metadata", "targets"):
    os.mkdir(os.path.join(tmp, name))
repo_dir = os.path.join(tmp, "repo")

sim = RepositorySimulator()
dump_repository(sim, repo_dir)

env = {
    **os.environ,
    "PYTHONPATH": os.getcwd(),
    "TUF_CRASH_METADATA_DIR": os.path.join(tmp, "metadata"),
    "TUF_CRASH_TARGETS_DIR": os.path.join(tmp, "targets"),
    "TUF_CRASH_REPO_DIR": repo_dir,
}
helper = [sys.executable, "tests/_crash_helper.py"]

def run(hook):
    result = subprocess.run(helper + [hook], env=env, capture_output=True)
    print(hook, "->", result.returncode, result.stdout.decode().strip())
    return result

# 1) complete refresh: creates generation 1
run("refresh")
print("pointer:", open(os.path.join(tmp, "metadata", "current")).read())

# 2) publish a new repository state, then die between rename and pointer swap
sim.targets.version = 2
sim.update_snapshot()
dump_repository(sim, repo_dir)
run("renamed")  # prints: renamed -> 18

# 3) inspect: pointer still points at generation 1, generation 2 is on disk
print("generations:", sorted(os.listdir(os.path.join(tmp, "metadata", "generations"))))
print("pointer still:", open(os.path.join(tmp, "metadata", "current")).read())

# 4) a new Updater adopts the complete generation 2 and reports targets v2
run("refresh")
print("pointer now:", open(os.path.join(tmp, "metadata", "current")).read())
PY
```

Expected conclusion: after the crash the directory is readable, the pointer and
the on-disk generation are each complete, and the next process adopts
generation 2 without any manual repair. The final JSON reports `targets: 2`.

## 3. A killed lock holder never blocks later processes

```bash
python3 -m unittest -v \
  tests.test_local_store.TestMultiProcess.test_lock_is_released_when_holder_dies \
  tests.test_local_store.TestMultiProcess.test_contending_writer_fails_explicitly
```

The first test starts a child that takes the lock, prints `locked` and then
sleeps forever; the parent kills it and verifies that a new process acquires
the lock in well under a second (the kernel drops the lock). The second test
verifies that a writer which cannot get the lock raises `LockTimeoutError`
instead of writing without it.

## 4. Damaged generation rolls back and is reported

```bash
python3 -m unittest -v \
  tests.test_local_store.TestStoreLayout.test_damaged_generation_rolls_back_and_reports \
  tests.test_local_store.TestStoreLayout.test_no_complete_generation_raises
```

The first test truncates the newest generation's `targets.json`, then asserts
that an `ERROR` log line reports the rollback, that the previous complete
generation becomes visible, and that the rollback is persisted. The second
removes the only generation's manifest and asserts
`IncompleteMetadataGenerationError`: no partial set is ever accepted.

## 5. Targets: no partial or unverified file is ever visible

```bash
python3 -m unittest -v \
  tests.test_local_store.TestCrashRecovery.test_crash_mid_target_download_does_not_publish_target \
  tests.test_local_store.TestMultiProcess.test_unverified_target_is_never_cached \
  tests.test_local_store.TestMultiProcess.test_cached_target_is_reverified
```

These assert that a killed download leaves no visible file, that a target
failing hash verification is not written at all, and that a corrupted cached
target is not reported as a cache hit.

## Single-process cost

The single-process test suite measures no timing thresholds (they would be
flaky), but the design keeps the added work minimal: one small pointer file read
when the local generation is first resolved, and one generation commit (a
directory rename plus one atomic pointer replacement) per successful refresh.
The complete existing test suite, including all ngclient workflows, runs in the
same range as before.
