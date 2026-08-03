# How to measure capability overhead on the FPGA

The method behind every number in `fpga-silicon-measurements-for-paper.md`. Read this
before producing or citing an overhead ratio.

**The one-line rule: the baseline must run bare-metal.** A Linux-hosted baseline is
wrong by 21 % and the error flatters us.

---

## 1. What is being compared

One kernel, `<rung>_kernel.h`, built **twice from identical source by the same clang at
the same `-O`**:

| half | build | harness |
|---|---|---|
| **capability** | `-target capstone64` + gp-captable silicon flags | domain entered by `cscall` (`build-ladder-fpga.sh`) |
| **baseline** | `-target riscv64`, no capability flags | bare-metal S-mode payload (`build-ladder-base-bare.sh`) |

`overhead = capability ÷ baseline`, reported separately for **cycles** and
**instructions**. Both halves bracket the compute only, so domain entry/exit is excluded
from both and needs no correction.

Two gates that must never be relaxed:

- **Static gate** — the build fails if a capability instruction reaches the baseline.
  Without it the "plain RISC-V" denominator could quietly measure capabilities too.
- **Oracle gate** — a rung counts only if it returns the value computed natively from the
  same source. A number from a run that computed the wrong answer is not a measurement.

## 2. Why the baseline is bare-metal

A Linux-hosted baseline services **timer interrupts inside the measurement bracket**.
Measured, not assumed, with a control kernel (`ctrsanity`) whose measured region is a
5-instruction register-only loop that both targets compile to the *identical* RISC-V
instructions:

| baseline | cycles | instret | cyc/iter | ratio vs capability |
|---|---:|---:|---:|---:|
| Linux | 728,727 | 509,178 | 7.287 | 0.824 (absurd: "capabilities faster") |
| **bare-metal** | **600,041** | **500,022** | **6.000** | **1.000** ✅ |
| (capability domain) | 600,309 | 500,033 | 6.003 | — |

Interrupt cost runs at **~14 cycles per instruction** against real code's ~1.8, so it
inflates the baseline's cycles *and* its CPI. That
**inflates the denominator, so it UNDERSTATES capability overhead** — the error runs in
our favour, which is why it survived so long. It also silently inverted the
"overhead is ABI, not enforcement" conclusion, because that argument turns on CPI.

**Rejected alternatives, and why:**

- *Repeat N times, take the least-disturbed pass.* Works below ~2 k cycles, partial to
  ~170 k, **useless above ~700 k** — for `ctrsanity` it recovered 7.290 against 7.287
  where truth is 6.000. `rv8_primes` (16.5 M cycles) can never be cleaned this way.
- *Calibrate and subtract.* Models the error instead of removing it, and the per-interrupt
  cost depends on the kernel's cache footprint.

## 3. Running a measurement

```bash
# baseline (bare-metal, no OS)
bash capstone/tests/rtl-smoke/build-ladder-base-bare.sh
FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)" \
  python fpga_driver/run_base_bare_fpga.py

# capability half
bash capstone/tests/rtl-smoke/build-ladder-fpga.sh <rungs...>
# DEPRECATED (UART-transfers each rung). Bake the rungs in instead and use:
#   BAKED_RUNGS="<rungs...>" python -m fpga_driver.run_baked_rungs_fpga
FPGA_URL=... LADDER_OPT=<level> python fpga_driver/run_ladder_perf_fpga.py
```

**`LADDER_OPT` must be set on the RUNNER, not just on a pre-build** — the runner rebuilds
by default and discards a pre-built set. Both halves take their per-rung `-O` from the
shared `ladder-rungs.spec`, and the runner hard-fails on a capability/baseline mismatch
(issue I-1, which once produced five bogus "silicon failures").

**Per-rung build knobs live in the spec, NOT on the command line (since 2026-07-28).**
`ladder-rungs.spec` has an optional 5th field of space-separated `KEY=VALUE` assignments
applied to the capability half only — today `DOMAIN_WINDOW=32k` (C-5) and
`LADDER_NO_RO_COPY=1` (C-4b). Before this, a whole-set sweep was impossible: applying the
knobs to every rung perturbs already-published rows (layout sensitivity — a 2026-07-26 A/B
where four added instructions flipped a passing rung), and applying them to none silently
builds the rungs that need them at 4 KiB with the broken copy path. That is why
`rv8_sha512` sat QEMU-verified but unmeasured, and why R-7 asked for exactly this. **Do
not reintroduce these as env vars** — a per-rung build property belongs in the one file
both halves read, same as `-O`. The baseline half discards field 5 deliberately.

**RETIRED 2026-08-03 — there are no transfers any more.** This section used to require
`burst=16` on the first attempt of a `fast_put`. UART delivery is now banned outright: bake
the rungs into the buildroot image and invoke them from the shell. `run_ladder_perf_fpga.py`
is DEPRECATED (it transfers); the sanctioned driver is
`fpga_driver/run_baked_rungs_fpga.py`, which runs a whole set in ONE boot with no transfer at
all. See `HOW-TO-LAUNCH-ON-FPGA.md` §"UART TRANSFER IS RETIRED".

**Throughput:** `LADDER_DISTINCT_VA=1` + `LADDER_ONE_BOOT=1` runs a whole sweep in one
boot instead of one boot per rung (R-3 is address-keyed). Validated measurement-safe: a
rung measured as 2nd domain matches its 1st-domain value to 0.03 %. Keep a known-good
control rung in any sweep using it — if the assumption ever fails, the symptom is a
silent hang that looks like a result.

## 4. Reading the output

Each rung is measured 16 times. The consumer keeps the pass with **minimum instret** and
reports how many passes **tied** at it, plus the cycle spread.

- **15/15 tied, spread 0** — bare-metal's normal signature. Trust it.
- **Few ties, large spread** — something is perturbing the run. Do not publish it.

**Byte-identical instret across two passes is NOT sufficient evidence of cleanliness.**
`beebs_cnt` satisfied that test on two separate Linux samples and still produced an
impossible 0.684× ratio: two passes can take the same number of interrupts and both be
contaminated.

## 5. How the bare-metal baseline is built

OpenSBI boots our S-mode program **in place of the Linux Image**. The firmware reuses the
known-good OpenSBI+FDT prefix of the existing `fw_payload` and substitutes our payload at
offset `0x200000` → `0x80200000` (confirmed by OpenSBI's own `Domain0 Next Address`).
The rung kernels are the **same** `ladder_base_kern.c` objects the Linux controller
links, so the measured code is unchanged; only the harness differs.

Details that are load-bearing:

- **`-mcmodel=medany`** — the payload sits above the low 2 GB that medlow's `lui/addi`
  can reach. Verified not to perturb the comparison: bare `beebs_prime` instret 12,562 vs
  Linux min-of-16 12,558 (0.03 %).
- **Console is direct ns16550a MMIO, not SBI.** The board reports `Runtime SBI Version:
  1.0`, so DBCN (SBI 2.0) cannot exist, and the legacy console is not built in — three
  board sessions produced the OpenSBI banner and then silence. Parameters come from the
  firmware's **device tree** (`/soc/uart@10000000`, `reg-shift=2`, `reg-io-width=4`; PMP
  Region05 grants S/U access), not from guesswork. QEMU's virt has the same chip at the
  same address with **shift 0**, so `UART_SHIFT` is a build parameter.
- **`.bss` is zeroed in `_start`** — the kernels rely on zero-initialised statics.
- **No interrupt is ever enabled.** No timer is programmed and no trap vector installed,
  so nothing can preempt a measurement; a stray trap hangs visibly rather than silently
  perturbing a number.
- Firmware is **2.1 MB vs 15.4 MB**, so the JTAG reload that dominates each boot is much
  faster — correctness and speed from the same change.

## 6. Validate off-board first

The payload runs unmodified under QEMU with real OpenSBI:

```bash
UART_SHIFT=0 OUT_DIR=/tmp/capstone/ladder-base-bare-qemu \
  bash capstone/tests/rtl-smoke/build-ladder-base-bare.sh
capstone/capstone-qemu/build/qemu-system-riscv64 -M virt -smp 1 -nographic \
  -kernel /tmp/capstone/ladder-base-bare-qemu/ladder_base_bare.elf
```

Expect 256 result rows over 16 rungs with every retval matching its oracle. **Do this
after any change to the payload.** Board time is the scarce resource, and three sessions
were spent on a console problem that QEMU plus the device tree would have caught for free.

## 7. Traps that have cost real board time

| trap | symptom | rule |
|---|---|---|
| `-O` set on pre-build only | runner rebuilds at its own default | set `LADDER_OPT` on the runner |
| grep for error strings | failing build reported as success | **gate on exit status** |
| capture index taken before reboot | successful run reported "0 UART chars" | console clears its buffer; fall back to the whole buffer |
| `puts_` converted, `putu_` not | labels print, **all numbers blank** | route every output path through one console |
| cleanup on a dead socket | board left **locked and powered on** | reconnect in `finally` before `power(False)`/`unlock()` |
| hand-maintained rung table | rung silently reports `--` | generate tables from `ladder-rungs.spec` |
