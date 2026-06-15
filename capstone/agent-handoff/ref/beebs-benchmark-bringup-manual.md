# BEEBS Benchmark Bring-Up Manual

This is the operational checklist for adding one or more BEEBS benchmarks to the
Capstone split host/domain benchmark path. It is written for coding agents that
need to produce the same kind of change set as the current Capstone BEEBS
bring-up work.

## Ground Rules

- Build benchmark domain code with the current repository's Capstone compiler:
  `$CAPSTONE_CLANG`, targeting `capstone64-unknown-elf`.
- Run benchmark domains through the current split host/domain runtime under the
  Capstone QEMU/OpenSBI/Buildroot environment, using the per-benchmark
  `run-beebs-*.sh` wrapper.
- Keep fetched BEEBS sources under `$CAPSTONE_TMP_ROOT`, normally
  `/tmp/capstone`. Do not vendor BEEBS sources and do not add submodules.
- Treat correctness markers as the success policy. Do not report benchmark
  performance numbers and do not tune for speed during bring-up.
- Add committed per-benchmark entry points even when using shared helper code.
- Use shared wrappers for ordinary BEEBS benchmarks. Add per-benchmark
  domain/host C files only when the marker ABI or host behavior genuinely
  differs.
- Do not add a broad permanent suite runner yet.
- If a candidate exposes a hard backend/compiler/runtime bug, unclear
  architecture semantics, or repeated failed runtime debugging where higher
  thinking is needed, stop and report the blocker instead of continuing the
  batch.

## Startup Checklist

From a fresh session, read:

1. `capstone/agent-handoff/README.md`
2. `capstone/agent-handoff/state/current-state.md`
3. `capstone/agent-handoff/state/current-next-step.md`
4. This file.

Then set up the shell:

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
mkdir -p "$CAPSTONE_TMP_ROOT"
```

Confirm the BEEBS source tree is available:

```bash
bash capstone/benchmarks/beebs/fetch-beebs.sh
```

The fetch script must place sources under `$CAPSTONE_TMP_ROOT/beebs-src`.

## Existing Helper Model

The common path has these reusable files:

- `capstone/benchmarks/beebs/beebs_simple_domain.c`
- `capstone/benchmarks/beebs/beebs_simple_host.c`
- `capstone/benchmarks/beebs/build-beebs-simple-capstone-common.sh`
- `capstone/benchmarks/beebs/build-beebs-simple-host-common.sh`
- `capstone/benchmarks/beebs/run-beebs-simple-common.sh`

A normal simple benchmark adds only three thin scripts:

- `build-beebs-<name>-capstone.sh`
- `build-beebs-<name>-host.sh`
- `run-beebs-<name>.sh`

`<name>` is the committed wrapper name and output stem. Prefer the BEEBS
directory name with hyphens preserved, for example `sglib-listsort`. The output
files are named `beebs_<name>_capstone.dom` and `beebs_<name>_host.user`.

The common Capstone build helper expects these variables before it is sourced:

- `BEEBS_BENCHMARK`: benchmark name/stem.
- `BEEBS_SOURCE_FILES_REL`: array of BEEBS source files relative to
  `$CAPSTONE_TMP_ROOT/beebs-src`.
- `BEEBS_EXTRA_INCLUDE_RELS`: optional array of BEEBS include directories
  relative to `$CAPSTONE_TMP_ROOT/beebs-src`.
- `BEEBS_STRIP_HOSTED_INCLUDES=1`: optional; strips simple hosted includes such
  as `<stdio.h>` and `<stdlib.h>` from copied sources.
- `BEEBS_DEFINE_NULL=1`: optional; injects a minimal `NULL` definition before
  compiling the copied source.
- `DOMAIN_OPT_LEVEL`: optional runtime override; defaults to `-O0`.

## Candidate Selection

Prefer candidates that:

- have deterministic inputs and outputs,
- have a real `verify_benchmark()` implementation,
- build from one or a small number of C files,
- avoid hosted libc dependencies,
- avoid floating point unless the cheap probe passes immediately,
- do not require large source adaptation.

Defer candidates that:

- have `verify_benchmark()` returning `-1`,
- need system calls or hosted file I/O,
- fail in backend instruction selection,
- produce the wrong correctness marker,
- require non-trivial source rewriting,
- expose runtime or OpenSBI/QEMU issues.

Record deferred candidates in
`capstone/agent-handoff/state/current-next-step.md` with the observed failure
class.

Useful source inspection commands:

```bash
find "$CAPSTONE_TMP_ROOT/beebs-src/src" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort
find "$CAPSTONE_TMP_ROOT/beebs-src/src/<candidate>" -maxdepth 2 -type f | sort
sed -n '1,240p' "$CAPSTONE_TMP_ROOT/beebs-src/src/<candidate>/<source>.c"
rg -n "verify_benchmark|initialise_benchmark|benchmark\\(" "$CAPSTONE_TMP_ROOT/beebs-src/src/<candidate>"
```

## Add One Simple Benchmark

Use this path when the benchmark can use `beebs_simple_domain.c` and
`beebs_simple_host.c`.

### 1. Create the Capstone Build Script

Create `capstone/benchmarks/beebs/build-beebs-<name>-capstone.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=<name>
BEEBS_SOURCE_FILES_REL=(src/<beebs-dir>/<file1>.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/<beebs-dir>)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
```

For multiple source files:

```bash
BEEBS_SOURCE_FILES_REL=(
  src/<beebs-dir>/<file1>.c
  src/<beebs-dir>/<file2>.c
)
```

Only set `BEEBS_STRIP_HOSTED_INCLUDES` or `BEEBS_DEFINE_NULL` when needed.

### 2. Create the Host Build Script

Create `capstone/benchmarks/beebs/build-beebs-<name>-host.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=<name>
source "$SCRIPT_DIR/build-beebs-simple-host-common.sh"
```

### 3. Create the Run Script

Create `capstone/benchmarks/beebs/run-beebs-<name>.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=<name>
source "$SCRIPT_DIR/run-beebs-simple-common.sh"
```

### 4. Make Scripts Executable

```bash
chmod +x \
  capstone/benchmarks/beebs/build-beebs-<name>-capstone.sh \
  capstone/benchmarks/beebs/build-beebs-<name>-host.sh \
  capstone/benchmarks/beebs/run-beebs-<name>.sh
```

### 5. Validate the New Benchmark

Build-only checks:

```bash
bash capstone/benchmarks/beebs/build-beebs-<name>-capstone.sh
bash capstone/benchmarks/beebs/build-beebs-<name>-host.sh
```

End-to-end runtime check:

```bash
bash capstone/benchmarks/beebs/run-beebs-<name>.sh
```

Expected output includes:

```text
beebs-<name>-host: correctness marker validated
__BEEBS_<NAME>_PASSED__
```

The run log is written to:

```text
$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-beebs-<name>.log
```

If QEMU times out before the Buildroot login prompt and the benchmark never
starts, rerun once before classifying the benchmark as failed. If the benchmark
starts and returns the wrong marker or crashes consistently, defer it and record
the failure class.

## When to Add Custom Domain or Host Files

Do not add custom `beebs_<name>_domain.c` or `beebs_<name>_host.c` just to carry
different strings. The shared host already accepts the benchmark name as an
argument.

Add custom files only when at least one of these is true:

- the domain must emit a different marker ABI,
- the host must validate a different protocol,
- the benchmark needs non-standard domain entry behavior,
- the benchmark requires a carefully documented source patch that does not fit
  the common helper's simple sanitization knobs.

If custom wrappers are needed:

1. Start from an existing custom benchmark such as `fac`, `fibcall`, or
   `insertsort`.
2. Keep the custom behavior narrow and documented in the script.
3. Add comments only where the behavior is not obvious.
4. Validate both the custom benchmark and one shared-wrapper benchmark afterward
   to prove both paths still work.

## Add Multiple Benchmarks at Once

Use batch mode to reduce token and runtime overhead. The recommended batch size
is 5-8 candidates.

### 1. Probe Candidates Before Committing Files

Work in `/tmp/capstone` first. For each candidate, identify:

- benchmark name,
- source files,
- include directories,
- whether hosted include stripping is needed,
- whether a `NULL` definition is needed,
- whether `verify_benchmark()` is meaningful.

Use temporary scripts or shell snippets under `$CAPSTONE_TMP_ROOT` to compile
each candidate against the common helper. Do not commit probe scripts.

Recommended quick classification:

- `PASS`: builds and `run-beebs-<name>.sh` validates the marker.
- `BUILD_FAIL`: compiler/backend/linker failure.
- `RUNTIME_FAIL`: QEMU boots, benchmark starts, but marker validation fails or
  the domain crashes.
- `INFRA_FLAKE`: QEMU fails before the benchmark starts; rerun once.
- `DEFER`: requires non-trivial source adaptation or bug investigation.

Do not debug `BUILD_FAIL` or `RUNTIME_FAIL` candidates inside a batch unless the
fix is obvious and local. Add cheap passers first, defer hard cases.

### 2. Add Only Passing Candidates

For every `PASS` candidate, add the same three committed scripts described in
"Add One Simple Benchmark":

- `build-beebs-<name>-capstone.sh`
- `build-beebs-<name>-host.sh`
- `run-beebs-<name>.sh`

Keep all passing simple benchmarks on the shared helper path unless they require
custom behavior by the rules above.

### 3. Validate the Batch

Run each new benchmark:

```bash
bash capstone/benchmarks/beebs/run-beebs-<name1>.sh
bash capstone/benchmarks/beebs/run-beebs-<name2>.sh
bash capstone/benchmarks/beebs/run-beebs-<name3>.sh
```

Then run the required regression set:

```bash
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-fac.sh
bash capstone/benchmarks/beebs/run-beebs-strstr.sh
bash capstone/benchmarks/beebs/run-beebs-ndes.sh
```

Also rerun one benchmark from the immediately previous batch, preferably the
latest simple-helper benchmark such as `run-beebs-expint.sh`, unless that exact
benchmark is part of the current batch.

### 4. Record Deferred Candidates

Update `capstone/agent-handoff/state/current-next-step.md` with:

- candidates added,
- candidates deferred,
- concise failure class for each deferred candidate,
- next recommended candidate pool.

Do not bury deferred-candidate notes in temporary logs.

## Documentation Updates

For every committed BEEBS addition or wrapper-policy change, update durable
handoff docs in the same change set.

Required updates:

- `capstone/agent-handoff/state/current-state.md`
  - Add each newly validated `run-beebs-<name>.sh` to the verified baseline.
  - Update the total/count wording if present.
  - Keep the shared-wrapper policy accurate.
- `capstone/agent-handoff/state/current-next-step.md`
  - Replace the completed milestone with the next concrete milestone.
  - List newly deferred candidates and why.
  - Keep the thinking-level rule and "do not vendor" rules intact.
- `capstone/agent-handoff/README.md`
  - Update the current verified baseline summary when the benchmark list changes.
- `capstone/agent-handoff/ref/testing-matrix.md`
  - Add new benchmark run commands to the benchmark regression list if the
    current matrix enumerates BEEBS wrappers.
- `capstone/agent-handoff/ref/capstone-agent-test-instructions.md`
  - Update practical command examples only if the workflow or common command set
    changed.

Do not commit manager-facing summaries, temporary investigation logs, or probe
artifacts. Those belong under `$CAPSTONE_TMP_ROOT`.

## Regression Requirements

Minimum validation for one added benchmark:

```bash
bash capstone/benchmarks/beebs/run-beebs-<name>.sh
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-fac.sh
bash capstone/benchmarks/beebs/run-beebs-strstr.sh
bash capstone/benchmarks/beebs/run-beebs-ndes.sh
```

Minimum validation for a batch:

- every new `run-beebs-<name>.sh`,
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- focused existing BEEBS regressions: `fac`, `strstr`, `ndes`,
- one benchmark from the previous batch, preferably `expint` unless already
  covered.

Additional validation:

- Run `bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh` if the change
  touched runtime harness behavior, QEMU/OpenSBI integration, or anything outside
  the benchmark wrappers.
- Run the broader HostCall/null_blk matrix if the change touches shared runtime
  code rather than benchmark-only scripts.

## Pre-Commit Review Checklist

Before staging:

```bash
git status --short
git diff --stat
rg -n "beebs_<name>_(domain|host)\\.c" capstone/benchmarks/beebs
```

Check that:

- no fetched BEEBS source was added to the repository,
- no `/tmp/capstone` artifact was added,
- scripts are executable,
- simple benchmarks use shared helpers,
- custom wrappers have a real reason,
- all success markers in run scripts match the host output,
- documentation reflects the new verified baseline and next step,
- test logs show marker validation, not only successful build.

Stage only the intended files. Do not stage unrelated local changes.

Use a full multi-line commit message. Suggested shape:

```text
Add BEEBS <names> benchmarks

Add Capstone domain, host, and run wrappers for <names>. These benchmarks use
the shared simple BEEBS domain and host helpers because their correctness-marker
behavior matches the standard wrapper protocol.

Update the handoff state, next-step guidance, and regression documentation so
future sessions know the new verified baseline and which candidates remain
deferred.

Validation: run-beebs-<name1>.sh passed; run-beebs-<name2>.sh passed;
llvm/test/CodeGen/Capstone passed; CoreMark passed; fac/strstr/ndes passed.
```

Do not add `Co-Authored-By` lines.

## Failure Handling

For a failed candidate, capture only the actionable facts:

- benchmark name,
- exact command that failed,
- failure class,
- first meaningful compiler/runtime error,
- whether the benchmark started under QEMU,
- whether the marker was wrong, missing, or never reached.

Store bulky logs under `$CAPSTONE_TMP_ROOT`. Put concise deferred-candidate
entries in `current-next-step.md`.

Stop and report instead of continuing when:

- the same failure repeats after simple local checks,
- the candidate appears to need backend compiler work,
- runtime behavior points at QEMU/OpenSBI/hostcall semantics,
- source adaptation would become benchmark-specific research,
- higher thinking appears necessary.

