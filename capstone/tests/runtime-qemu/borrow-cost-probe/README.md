# borrow-cost-probe — borrow-path cost measurement (task-014)

Paper deliverable 2: *the capability-mediated borrow path stays close to raw
pointers and well below the copy baseline.* One domain measures the **dynamic
instruction count** of three variants of the same borrow-one-word boundary
operation — **raw pointer**, **capability borrow** (`mrev`+`delin`+access+
`revoke`), and a **defensive copy** — on the QEMU functional model.

**Results and the methodology caveat live in `RESULTS.md`.** Headline: borrow
costs +4 instructions over a raw pointer (payload-independent), and is 5.7×–21.7×
cheaper than a 256 B–1024 B defensive copy (copy is O(payload); borrow/raw are
O(1)). This is a functional-model **instruction-count proxy, not cycle-accurate
timing**.

## Files

| File | Role |
|---|---|
| `borrow_cost.c` | domain payload (Capstone clang, `-O2`): the three measured loops, bracketed by the `csrdicount` readout. |
| `borrow_cost_probe_guest.c` | controller (buildroot gcc): creates + transfers the LINEAR arena, makes the CALL. |
| `borrow_cost_probe.h` | shared constants (sizes, DPI codes, counter slots). |

## Run

```bash
capstone/tests/runtime-qemu/run-borrow-cost-probe.sh
```

Needs the `rootfs.ext2` lock (one QEMU boot; announce in `COORDINATION.md`). The
run boots with `-icount shift=0` (deterministic instruction count) via the
additive `--qemu-extra-arg` passthrough in `run-domain-smoke.py`, greps the
Capstone debug-counter dump from the serial log, and prints per-op counts and
ratios. `build-borrow-cost-probe.sh` also emits `asm/borrow_cost.s` for the
static cross-check.

## How the count is taken

`csrdicount rd` (a Capstone debug op added to `capstone-qemu`, opcode `0x5b`
funct3 `0x1` funct7 `0x48`) reads QEMU's raw retired-instruction count (icount)
into `rd`. Under `-icount` one tick == one retired instruction, so a delta across
a code region is that region's exact dynamic instruction count. The domain reports
results through the existing `csdebugcount`/`csdebugcountprint` counters, which
land in the serial log. See `RESULTS.md` for why the monitor/ioctl path is
excluded (the paper's mechanism is the inline capability ISA sequence; the
software-monitor syscall scaffolding is an emulation artifact).
