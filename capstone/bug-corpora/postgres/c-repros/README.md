# PostgreSQL 17.5 defects reduced to C, with no server

Five memory-safety defects in PostgreSQL's frontend and shared code -- libpq, pg_dump, ecpg,
saslprep, the basebackup tar streamer -- each a `case.c` that drives the **real upstream function**,
compiled verbatim from the 17.5 pin (`shared/upstream_*.c`), with no server in the process. The
defective object comes straight from `malloc`: these are the PostgreSQL corpus's non-nested cases.
Siblings: [`../sql-repros`](../sql-repros) (defects reached through SQL on a running server) and
[`../mmgr-repros`](../mmgr-repros) (defects in the memory managers, replayed).

The layout is the contract in [`../../SCHEMA.md`](../../SCHEMA.md): one directory per case, a
`case.c`, a `case.json` of claims and a `PROVENANCE.md`.

## Arms, and what each one is

| arm | configuration (`tools/arms.json`) | what it is |
|---|---|---|
| `spatial` | `app-level0` | application domain, SDK heap level0: malloc bounds each object; free revokes nothing |
| `virtual-malloc` | `virtual-mallocng` | the virtual Capstone profile (`capstone/runtime/virtual`): the case runs as a Linux process under `capstone-vexec`, malloc is musl mallocng run locally with exact bounds and lifetime retirement on free |
| `cheribsd-revocation` | -- | stock CheriBSD purecap, revocation at the platform default (`shared/run-cheribsd.sh`) |
| `host-asan` | -- | the host build under ASan |

With no nested allocator in these programs, the protected system allocator -- virtual mallocng --
is the program's whole protected configuration. The physical Sublet heap is not used (project
decision, 2026-10-10).

**The arm recorded as `sublet` on 2026-10-06 was not the Sublet heap, and is dropped.** It ran on
the SDK that `ports/postgres/app/build-domain.sh` builds for the server's `PGSU_NESTED=sublet` arm.
Both PostgreSQL SDKs keep the default `CAPSTONE_APPLICATION_HEAP level0`
(`runtime/application/CMakeLists.txt`); the sublet one only adds a grant for the server's context
pools, which these programs never use. Its five catches -- at the same pcs as the `spatial` run --
were level0 bounds under a Sublet label. The run stays in `results/sublet-20261006-161014` as the
record of what was measured; no verdict is taken from it.

## Building and running

    # once per arm: an application SDK built with the arm's heap
    bash shared/build-domain.sh <SDK> <OUT>          # cases, controls.dom, build.json (the SDK's heap)
    python3 shared/run-arm.py --arm spatial --state <capstone-vm state> --bindir <OUT> \
        --llvm-bin <compiler>/bin

On the virtual profile the SDK is the one a virtual application build made (for example
`<build-virtual.sh postgres OUT>/image/sdk`), and the run takes `--virtual-kit <platform>` instead
of `--state`: every control and case then runs in one boot of the qualified virtual platform
(`tools/virtualvm.py`, `capstone/runtime/virtual/run-staged.py`).

`run-arm.py` reads the heap from `<OUT>/build.json` and refuses an arm whose configuration needs
another one. Before any case it runs the configuration's controls from `controls.dom` (a write one
past a malloc'd object, a read after free; `shared/controls.c`), then each case, and reports one
Observation per run to the shared judge (`tools/verdicts.py`, SCHEMA.md "Verdicts"). A fault counts
as the defect's only in the function the case names before its access (`expect_fault_in`) or in a
`fault_sites` entry its `case.json` justifies; the pc is resolved from the image's own symbols. The
bundle goes to `results/<stamp>-qemu/<arm>/`; console logs stay outside the repository.

## Results on record

`results/matrix.tsv` and the three `results/*-20261006-*` directories are the 2026-10-06 runs,
scored by the runner of that day: any fault after the case began counted as detected. On those
runs the pc of every `spatial` fault is recorded but was never compared with the function the case
names -- case 0's fault at `0xc02020f0` lies below `pqGetnchar@0xc020f130` -- so they are not
re-judged here as attributed.
