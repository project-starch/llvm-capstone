# The SQLite 3.22.0 temporal-bug corpus on R-43 v2 (`caplifive_r43_8f6a0af.bit`), batch 1, from 2026-10-01

The lead asked for one of the newly ported test sets on the FPGA. Most of them cannot run there:
- the application ports (mruby, Perl, CPython, PostgreSQL, SQLite 3.22 as a program, tshark, FFmpeg) and the
  new suites (delegated probes; the signals, sockets and threads contracts; the heap gate) run on the
  delegated runtime;
- that runtime depends on the supervised CALL extension, which only capstone-qemu implements
  (`current-state.md`: "Existing FPGA hardware does not implement the new supervised CALL extension").

The SQLite 3.22.0 temporal-bug corpus (`ports/sqlite/repro322`, PR #171/#174) is the exception. Each case is a
freestanding domain driven by the bare-HostCall SQLite host, the same model as our SQLite board runs. This
folder runs it on silicon.

**What the corpus is.** It is the CONTROL arm of a temporal-safety study. Each case drives SQLite to a
freed-then-used path on unprotected Capstone: SQLite's own memsys5 and lookaside, nothing revoked.
- On the emulator, 14 rows return (`<tag> NOTRAP done`) and 4 fault.
- The PR's own reading: a FAULT is memsys5's integer freelist links overwriting a pointer field of the freed
  block, so that field loses its tag. It is not temporal enforcement.

**Which rows, and why.** Batch 1 is the rows the native ASan comparison of the #171/#174 gate confirmed, or
could not reach, and that need no extra translation unit:
- returning rows 8 (wschema), 20 (blobwrite), 25 (mem5design), 5 (blobclose), and #174's jsoneachstatic and
  jsoneachroot;
- faulting rows 15 (backupattach) and 4 (detachtrig);
- plus the 3.22.0 base domain, `sqlite_capstone_domain.c` itself, which has passed only the silicon-config
  QEMU gate (`ports/sqlite/README.md`) and has never run on silicon.

Left out, with the reason:
- rows 1, 3, 6, 11, 12, 13, 17 and 22: their reachability is unproven even on the emulator (native ASan stays
  silent, and they print no probe; see the gate comment on #171);
- rows 19 (expertrem) and 24 (spellfixoom): they need `sqlite3expert.c` and `spellfix.c` as extra
  translation units, which the one-TU silicon build cannot take as they are;
- row 10 (fts5inplace): it needs the fts stubs in the same TU, and floating point.

Rows 19, 24 and 10 are batch 2.

## Images

Built by `build-images.sh` here in the SILICON config (`build-sqlite-silicon.sh`: gp-captable, one
translation unit, interp glue). The corpus's own build (`build-sqlite-row322.sh`) uses `start.S` + `link.ld`,
which the emulator runs only with a fabricated gp.
- Same case sources and the same adapted 3.22.0 amalgamation (`adapt-sqlite-322.sh`, sqlite3.c sha256
  `b446615b…`). The corpus group's feature defines go through `DOMAIN_EXTRA_DEFS`, which the silicon build
  appends after its own SQLite defines and trims.
- Compiler: the frozen dev build with #147, `clang version 22.0.0git (… 612b3ec514c0)`, library manifest
  `18c0d1b3` (the manifest the #170/#171 gates verified). `SHA256SUMS` lists the nine images.

Each image has its own entry VA, 4 MiB apart, none at k800's 0x10000. That uses a new `DOMAIN_BASE_VA` knob
in `build-sqlite-silicon.sh`, added in the same change as this folder. Before it, every SQLite image entered at
0x10000, so a SQLite boot could carry neither a control nor a second image. The knob was checked two-sided:
- default: the rebuilt image is byte-identical to the one built before the edit (wschema `ab3ce7cf…`, entry
  0x10000);
- `0x410000`: the entry moves, and the link script carries `0x410000 + <goff>`;
- the substitution check refuses a reworded `link-gpfree.ld` on both branches.

## Emulator record (the predictions)

Each image ran under the main clone's capstone-qemu (`deb7d75756`) with `CAPSTONE_GP_FABRICATE=0`, the board's
own SQLite host (`sqlite_host.user`, `2c9e82d101b48160`) and the shared images. The seven returning images ran
in one boot in board order. Each faulting image ran alone. A host built from the QEMU buildroot gave
byte-identical output. A QEMU record for each image is in `~/capstone-artifacts/qemu-pass/<sha256>`, as
preflight C16 requires.

| cell | row | emulator reading of THIS image |
|---|---|---|
| base322 | - | `row name=alpha value=11`, `beta 22`, `gamma 33`, `__CAPSTONE_SQLITE_EXTENDED_PASSED__`, `__CAPSTONE_SQLITE_MEMORY_PASSED__` |
| wschema | 8 | CONTROL insert rc=0; the poisoned `insert rc=11 (database disk image is malformed)`; `wschema NOTRAP done` |
| blobwrite | 20 | `blobwrite write rc=0`, `blobwrite NOTRAP done` |
| mem5design | 25 | `before=2779054080 after=4294967295 (in-band metadata overwrote caller data)`, `mem5design NOTRAP done` |
| blobclose | 5 | `blobclose NOTRAP done` |
| jsonstatic | #174 | `cross_rows=6`, `max=[3,4,5]`, `rows=1`, `jsoneachstatic NOTRAP done` |
| jsonroot | #174 | `cross_rows=6`, three `root=$.bb` pairs, `max={"bb":[3,4,5]}`, `rows=1`, `jsoneachroot NOTRAP done` |
| backupattach | 15 | **cause 24**, image VA 0x2016c64 = `sqlite3BtreeUpdateMeta+0x48`: `ldc a1,0x80(a1)` through a pointer read from the stale object (PR's emulator site: same function, `+0x94` in its own build) |
| detachtrig | 4 | **cause 24**, image VA 0x24b7d7c = `sqlite3DropTriggerPtr+0x154`: `ldc a3,0(a1)` (PR's site: same function, `+0x268`) |

The fault VAs were symbolized by hand: VA = entry + (fault pc − runtime base), where the runtime base is the
first domain pc rounded down to 4 KiB. `fault-locate.py` assumes an image at 0x10000.

## Boots

`drivers/board-r322.sh BOOT=<n>`, with `cells.tsv` here. Common to both boots:
- resident bitstream `caplifive_r43_8f6a0af.bit` and `REFUSAL_RECORD=1`; monitor `2dcd3a5`; FPGA buildroot
  `d04bd83`; host `2c9e82d1`;
- the stock k800 control `b2d60e525f807ea4` at 0x10000 runs first, and the boot is void if it does not return 4;
- at most one FAULT cell per boot, and it runs last (M-1: a fault wedges the core).

| boot | cells, in order |
|---|---|
| 1 | k800, base322, wschema, blobwrite, mem5design, blobclose, jsonstatic, jsonroot, **backupattach** |
| 2 | k800, base322, **detachtrig** |

## Pre-registered readings

- **k800** returns 4.
- **Every returning cell** prints, on silicon, the emulator's lines above, including the domain's own values
  (`insert rc=11`, `cross_rows=6`, mem5design's `after=4294967295`), and returns.
- **Each FAULT cell** wedges with **cause 24** in the trap log, at the same image VA (the image is the same
  bytes). The refusal record reads EMPTY: nothing is revoked in the control arm, so cause 25 is not expected.

The informative readings are divergences from the emulator:
- **a FAULT cell that returns** (`NOTRAP done`): the RTL keeps a capability tag that memsys5's integer freelist
  stores clear in the emulator. That is a tag-clearing difference between the RTL and QEMU, to be investigated;
- **a returning cell that faults:** silicon refuses an access the emulator's control arm runs through. Read the
  cause, the trap PC and the refusal record before saying which;
- **cause 25 or a LATCHED refusal record anywhere:** a refusal with nothing revoked. Stop and investigate;
- **base322 not completing:** SQLite 3.22.0 does not yet run on silicon, independent of the corpus. The cells
  after it are then unread, not failed.

**Not established by these boots:**
- N = 1 per cell;
- this is the control arm only: nothing is revoked, so nothing here measures Sublet or temporal enforcement;
- rows 1, 3, 6, 11, 12, 13, 17 and 22 are not run;
- batch 2 (rows 10, 19, 24) is not built.
