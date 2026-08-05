# A ~1.6 MB pure-capability domain hangs after ANY perturbation of its image

**Status: reproducible board-vs-QEMU divergence. NOT root-caused, NOT minimised.**
Handed over as a reproducer plus a list of eliminated variables, not as a diagnosed defect.

## The observation

Two SQLite domain images differing by **one dead, never-called, empty C function** appended to
`sqlite_capstone_domain.c`.

| | QEMU | board (`caplifive_fixed_forward.bit`) |
|---|---|---|
| `uc.dom` stage 11 | `obs=1517161237` correct | `obs=1517161237` correct — 5 observations |
| `dp0.dom` stage 11 | `obs=1517161237` correct | **never returns** — 2 observations |

Stage 11 executes **only** `sqlite3Strlen30("capstone_probe_string")`. It never calls the added
function and never touches anything the diff changes.

The hang is **silent**: no trap reported, no marker, and the core keeps servicing the console.

## Why this is hard, and what it is not

Nine structurally different perturbations of `uc` were built and run. **Every one hangs.** Only
unmodified builds (`uc`, and the independently-built `f10`) return. Each of the following was
tested directly and **excluded** — please do not re-derive them:

| # | variable | how it was excluded |
|---|---|---|
| 1 | `.gct` (static capability table) size/contents | `dvar` (one dead **integer** global) has `.gct` **unchanged** at 108 bytes and hangs |
| 2 | number of globals / carves | synthetic scan 8 to 208 carves all correct; `dvar` at 182 carves hangs |
| 3 | image size | `sz2048`/`sz8192`/`sz16384` are **all exactly** 1,624,216 bytes (identical to `dp0`) and all hang — `.text` slack absorbs the padding |
| 4 | address of the executed code | **PARTIAL — do not treat as a clean exclusion.** `sqlite3Strlen30` is at the same address `0x16afc` in both, but it *calls* `strlen`, which moves, and an immediate inside it differs (`addi a1,a1,0xdc` vs `0x100`). The executed instruction stream is **not** identical. |
| 5 | the amalgamation rewrite | `dp0`'s `sqlite3-capstone.c` is **byte-identical** to `uc`'s |
| 6 | run position / boot health | `uc:11` returns at position 1; `dp0:11` hangs at position 1 **and** at position 4 behind a returning control |
| 7 | revocation-node pool exhaustion | debug mux: `rev_node_head` = 221 of 1021, `overflow=0` |
| 8 | capability bounds representability | rebuilt at `SQLITE_HEAP_SIZE=4096` so **every** carve is representable; still hangs |
| 9 | operand forwarding | this bitstream carries the fix (`capstone-ariane 7aac52f93`) |

**The debug mux is not discriminating here.** Read on the *passing* run it gives byte-identical
registers to the hanging run: `sw=255` `0x8f` (`trap_seen=1 mcause=15`), `sw=224` `0xff`,
`sw=225` `0xd5` (`wait_rev_res=1`, `mem_wait=1`, `stall_issue=1`, `privM=1`). Those are
background Linux state, not a wedge signature. Any future mux reading must subtract this
baseline.

## What the QEMU result does and does not license

It **does** rule out a platform-independent compiler/glue defect: a miscompiled `dp0` would fail
under QEMU too, and it does not.

It does **not** establish a hardware bug. QEMU is our own model and is permissive where the RTL
is not, and that asymmetry has produced false conclusions on this project before. The board may
be behaving correctly while our software relies on something it does not guarantee, and the
difference could be timing rather than function.

## Reproducing

```bash
source capstone/tests/capstone-test-env.sh
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"
export FPGA_FW=.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
bash capstone/tests/fpga-repros/S01-image-perturbation-hang/run.sh
```

`run.sh` builds both images, runs the QEMU differential (both must return `obs=1517161237`),
then runs the board pair in **one boot** with `uc` first as a live control.

**REPRODUCED = `uc` returns and `dp0` does not.** If both return, the divergence has gone —
check the resident bitstream before concluding anything.

## Ordering rules that make the verdict valid

* Run `uc:11` **first in the same boot**. Without a returning control the boot is void; that
  rule has already invalidated results here once.
* At most one expected-hang per boot, last — a hang takes the rest of the boot with it.
* Verify **both** `.dom` files are byte-present in the initramfs. The shipped freshness gate
  only checks the canonical `sqlite_silicon.dom`, so a run can otherwise test a stale image.

## A cheaper probe than the hang: sporadic wrong `strlen` results

Stages that **return** are already wrong, and a wrong number is far more tractable than a hang:

| stage | board | expected | QEMU |
|---|---|---|---|
| 13 | **15** | 36 (5+8+11+12) | 36 correct |
| 16 | **124** | 128 (128×5 & 0xff) | 128 correct |

Stage 16 calls `strlen` on the **same** literal `"alpha"` 128 times and totals 636, i.e. **4 of
128 calls returned 1** — **sporadic (~3%), not length-dependent**.

At `-O0` (the SQLite default, `build-sqlite-silicon.sh:41`) `strlen` re-loads the string
capability with `ldc` from a stack slot on **every iteration**; a result of exactly 1 is what a
failed second reload would produce. `-O1` would keep it in a register, but `-O1` **cannot
build** — see **C-17** (`i128 SELECT_CC` not selectable, itself a recurrence).

Inferred but **not established**: wrong `strlen` → wrong hash in `sqlite3InsertBuiltinFuncs` →
corrupt chain → the stage-10 hang.

**Unresolved conflict:** `SILICON-BLOCKER.md` §0a8 records stage 13 returning **36 CORRECT**
after the unaligned-copy fix. `f10` returns 15 today. Re-run on a current build first.

## What would help most from the hardware side

A waveform or simulation trace of `dp0` stage 11 around the hang. Every software-visible
observable we have is either identical between the two runs, or changes the outcome when we try
to read it. Specifically: what is the core waiting on, and does any FSM enter a state it cannot
leave?

Also worth knowing: **does the divergence survive on `caplifive_65536_nodes.bit`?** If it
disappears there, the answer was revocation-node related after all, despite `overflow=0`.
