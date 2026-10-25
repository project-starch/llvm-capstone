# Lifetime controls at SQLite's memsys5

Not defects. Three controls from the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(53325bba0d5c). The security table shows that the memsys5 lifetime patch rejects stale accesses.
These controls check the other half: that it leaves permitted accesses alone. Each control drives memsys5
through its own API: `sqlite3_malloc` and `sqlite3_free` over `SQLITE_CONFIG_HEAP`. The heap is the
security corpus's: `repro322_common.h`, one 1 MiB static array, 64-byte minimum.

| control | image | what it does | expected, both arms |
|---|---|---|---|
| `control-parent` | `control-parent.dom` | frees A, then reads and writes A's former range through `mem5.zPool + offset`, the reference memsys5 itself uses (`MEM5LINK`) | completes. The read returns memsys5's own free-list link `{next = previous head, prev = -1}` and A's bytes after it. The write is restored before memsys5 runs again, and memsys5 then hands A's block out again |
| `control-sibling` | `control-sibling.dom` | allocates A and B, frees A, then reads B, writes B, and reads B back | completes; B intact |
| `control-successor` | `control-successor.dom` | frees A, then allocates C of A's size until memsys5 returns A's start (at most 64 requests, each kept), then writes and reads C | completes; C's start equals A's |
| | `control-successor-stale.dom` | the same, then one read through the kept reference to A, at the probe `stale-A-read`. A separate image, so it is a separate run | protected: fault, cause 25, at that probe. Release: completes, and the read returns C's bytes |

The two arms are the corpus's (`ports/sqlite/repro322/build-virtual.py`):
- **release**: `--memsys5`, memsys5 as SQLite ships it;
- **protected**: `--sublet`, with `ports/sqlite/sublet/sublet-3220000-memsys5.patch` applied.
  Under the patch every block is a CDERIVE child of `mem5.zPool`, and `sqlite3_free` revokes it
  (CREVOKE).

## Probes

Every access to a block goes through `PROBE_READ` or `PROBE_WRITE` (`control_common.h`). Each
probe is written on one source line, and it prints `PROBE <label> <op> <file>:<line> addr=… len=…`
before the access and `PROBE <label> done` after it. The access sits in the macro body, so a
faulting pc symbolizes (`-g`, `llvm-symbolizer`) to that same file:line. Each prediction the
program can test itself is printed as a `CHECK <what> ok|FAILED` line.

## Build and run

    build-controls.sh <virtual SDK> <amalgamation dir> <out>/release   release
    build-controls.sh <virtual SDK> <amalgamation dir> <out>/protected protected
    run-controls.py --builds <out> --kit <platform kit> --llvm-bin <dir> --work <new dir>

`build-controls.sh` stages the engine with the corpus's builder, unchanged. That builder copies
the amalgamation, applies the FTS fixes, and on the protected arm applies the patch. Each control
is then compiled as one translation unit that `#include`s the staged `sqlite3-capstone.c`, because
`mem5` is a static of the amalgamation. The flags are the corpus's flags for a core-group case,
plus `-g`, at `-O0`.

`run-controls.py` is a local runner. The corpus has no Capstone runner of its own; engine-repros'
`shared/run-virtual.py` attributes a fault by function and has no probe lines. This runner boots
each image in its own VM through `runtime/virtual/run-staged.py`, one hart, exact bounds. It then
decides from the log:
- **completed**: returned, exit 0, no fault line, and every probe `done`;
- **fault at the probe**: cause 25, the last probe is `stale-A-read` with no `done`, and the fault
  pc symbolizes to that probe's line.

Results: [`../results/2026-10-11-controls/`](../results/2026-10-11-controls/).
