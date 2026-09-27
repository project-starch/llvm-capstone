# Four-configuration application memory study

This directory organizes known application benchmarks using the existing
persistent-guest runners. `catalog.json` records candidate suites, source pins,
application scope and qualification gaps. `study.py` fixes the workload matrix,
emits runner points from checked local artifacts, and reports the complete
attempt denominator. It does not implement another VM manager or vendor suites.

The initial catalog has six applications and eight candidate suites. No suite
is yet qualified for a four-configuration published comparison. Previous
FFmpeg/mruby discovery results remain useful but are not renamed as standard
benchmarks. The [research and execution plan](../../docs/plans/application-benchmark-study.md)
explains benchmark selection, measurement layers and the rollout.

## Comparisons

| Arm | Outer heap | Internal allocator | Process revocation policy |
|---|---|---|---|
| `capstone` | Shared SDK level0 | Original application implementation | Existing Capstone runtime |
| `capstone-sublet`, profile `nested` | Same level0 | Application's Sublet integration | Existing Capstone runtime |
| `capstone-sublet`, profile `outer-malloc` | Sublet malloc | Original application implementation | Existing Capstone runtime |
| `cheribsd-revocation-on` | Default CheriBSD malloc | Original application implementation | Explicitly enabled |
| `cheribsd-revocation-off` | Same binary and malloc | Same implementation | Explicitly disabled |

Each plan has exactly four arms and exactly one profile. The nested profile
addresses the internal-allocator paper question. The outer-malloc profile is
a separate comparison continuing the earlier experiments. Neither means
disabling the Capstone ISA. Level0 is the SDK's arena allocator, not jemalloc;
cross-platform differences include allocator policy and OS/runtime differences.

CheriBSD uses one kernel, libc, application binary and inputs for both arms.
Only `_RUNTIME_REVOCATION_ENABLE=1` versus `_RUNTIME_REVOCATION_DISABLE=1`
changes. The shared runner starts each study process with a clean environment,
keeps kernel defaults unchanged, records the effective environment, and checks
`malloc_revoke_enabled()` at every measurement phase. Other allocator-policy
overrides are rejected. The legacy `cheribsd-default` arm still inherits default
policy without these switches. [Implementation at the measured source pin](https://github.com/CTSRD-CHERI/cheribsd/blob/88f39900c32928d807dba245fba138808c666f34/lib/libc/stdlib/malloc/mrs/mrs.c#L1382).

## One plan, existing runners

Source the project environment before test/build/run commands. Plans, fetched
sources, build trees, local bindings and raw output belong under
`$CAPSTONE_TMP_ROOT`. Curated evidence and benchmark definitions belong here.

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/study/study.py list
python3 capstone/experiments/study/study.py plan \
  --profile nested --suites sqlite-speedtest1 cpython-pyperformance mruby-upstream \
  --repeat 3 --seed 20260927 --out "$STUDY_ROOT/plan.json"
```

The default variant executes upstream work counts. For size/retention/epoch
experiments, pass `--matrix matrix.json`. The matrix must explicitly cover every
selected suite case; each case maps variant names to work parameters:

```json
{
  "mruby-upstream/bm_so_lists.rb": {
    "small": {"list_size": 1000, "iterations": 30},
    "reference": {"list_size": 10000, "iterations": 300}
  },
  "mruby-upstream/bm_so_mandelbrot.rb": {
    "reference": {"width": 600, "height": 600}
  },
  "mruby-upstream/bm_ao_render.rb": {
    "reference": {"work": "upstream-default"}
  }
}
```

Nondefault parameters need a recorded workload adaptation. Changing the matrix
creates a different plan. Freeze the qualified plan and bindings before the
main campaign; do not choose the largest favorable endpoint after plotting.
Keep discovery and confirmatory campaigns separate.

## Qualified local bindings

A binding maps `suite/case@variant` to one reviewed workload contract. All four
arms are required before `points` emits any member of that comparison. Missing
bindings stay `unqualified`; an invalid supplied binding is an error. The
binding format is:

```text
{
  "suite/case@variant": {
    "source": <exact source object from catalog>,
    "application_version": <matching application version>,
    "profile": "nested" | "outer-malloc",
    "parameters": <exact parameters from plan>,
    "adaptation": <description of unchanged and adapted upstream behavior>,
    "oracle_reference": <preserved native reference stdout file>,
    "oracle_reference_sha256": <its SHA256>,
    "oracle": {"expected_stdout": <exact output>, "expected_phases": [...]},
    "input_sha256": {<input role>: <SHA256>, ...},
    "arms": {
      <each of the four arm IDs>: {
        "binary": <host binary path>,
        "build_manifest": <existing build manifest path>,
        "inputs": {<input role>: <host path>, ...},
        "resources": <pool/grant, tables, observer, stack and metadata accounting>,
        "qualification_evidence": <adapter correctness result identity>,
        "point": <existing application runner point with argv/environment/files>
      }
    }
  }
}
```

Build manifests identify the application, binary SHA256, `allocations=true`,
and `allocations_sha256`. Capstone manifests also carry `heap` and `nested`;
the latter is `none` or the application ID whose internal allocator is integrated.
The existing builder supplies this for CPython and mruby. Other integrations
need an application adapter before receiving that identity. Never manufacture
a manifest that labels an allocator replay as an application build.

Both CheriBSD arm entries share the same binary, manifest, argv, environment,
files, resources and qualification evidence. The runner supplies the policy
switch. Input hashes are checked across all arms; output and phase oracles are
common. The checks verify identities and declared configuration, not that a
manually described adapter is semantically correct. Qualification evidence must
include native reference execution and a review of workload/allocator boundaries.

```sh
python3 capstone/experiments/study/study.py points \
  --plan "$STUDY_ROOT/plan.json" --bindings "$STUDY_ROOT/bindings.json" \
  --platform capstone --repetition 0 --out "$STUDY_ROOT/capstone-r0.json"
python3 capstone/experiments/applications/run.py \
  --state "$CAPSTONE_VM_STATE" --points "$STUDY_ROOT/capstone-r0.json" \
  --repeat 1 --out "$STUDY_ROOT/capstone-r0"
```

Use `--repeat 1` on the existing runner: each emitted point already represents
one planned repetition. For CheriBSD, emit `--platform cheribsd` and use
`applications/cheribsd-run.py` with its existing `--sdk/--rootfs/--disk/--port`
options, or `--key` to attach to an owned persistent guest. Use the existing
interpreter environment with pexpect. Both policy arms run in one guest boot.
Do not run an empty point list. Respect the shared QEMU lock; stop only an owned
idle VM when switching platforms, and restore it in a `finally` block.

Repetition blocks alternate platform order. Within each block, a seeded order
interleaves both arms and workloads. Preserve that order when emitting each
platform/repetition file. VM changes may be batched when boot cost demands it,
but record that departure from the planned platform order. No automatic retries
or hidden restarts are permitted. Independent cells survive an ordinary
application fault; lost VM control or failed cleanup stops the platform batch.

Pass previous `runs.jsonl` files with `--runs` to emit only unattempted cells.
Failures remain attempted; they are not retried by resume. Duplicate attempts,
foreign plans, changed binding identities and changed local artifacts are rejected.
`study.py status --plan ... --bindings ... --runs ...` lists pass/failure/pending/
unqualified counts for every workload and arm. Unrecorded cells remain pending;
they cannot disappear from a success denominator. A lost runner between launch
and writing its record needs manual reconciliation before resuming that cell.

## Current verification and next integration

The planner has twelve tests for pairing, matrix identity, missing controls,
scope, input/observer/oracle identity, and preservation of failed attempts.
Six CheriBSD runner tests cover false passes and explicit policy selection.
The real policy smoke uses existing FFmpeg/mruby discovery images, not known
benchmark results: both on/off pairs and a subsequent on process pass in one
guest, with the expected state at every phase. See [the checked policy result](results/20260927-policy.json).
The [external evidence archive](results/20260927-archive.json) preserves raw
attempts, commands and the tested sources, with guest credentials excluded.

Next: qualify mruby's upstream workloads, SQLite speedtest1 and the selected
pyperformance bodies, including internal allocator counters. The existing
runner contract currently expects textual stdout; binary-output benchmarks
(such as Mandelbrot PBM output) need exact byte-hash oracles before qualification.
PostgreSQL's structured SQL oracle and FFmpeg's broader CLI need corresponding
shared runner/build support. No full standard suite is claimed to pass yet.
