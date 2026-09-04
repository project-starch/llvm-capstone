# Controlled A/B: four instructions flip a passing rung — and coremark hangs, not stalls

**Date:** 2026-07-26 · **Lane:** B · Two board sessions (4 boots total).
Board powered off + unlocked both times.

## 1. The A/B, now controlled

Two builds of the **same rung, same kernel, same compiler, same flags**, differing
only in `domain_main`:

| variant | `domain_main` | retval | oracle | cycles | verdict |
|---|---|---:|---:|---:|---|
| `beebs_prime` | + 2 `csrr minstret`, + 2 stores to `res[65]`, + 1 store to `res[64]` | 1087631800 | 582955588 | 47,952 | ❌ **wrong** |
| `beebs_prime_noins` | `mcycle` only (pre-26-07 body) | 582955588 | 582955588 | 47,780 | ✅ **correct** |

`beebs_prime_noins_fpga_app.c` is a first-class rung (`-DLADDER_NO_MINSTRET`), so it
passed the same QEMU oracle gate and the same build path as its twin.

**Four instructions, none of them inside the computation, flip a scalar rung from
correct to miscomputing on silicon.** The wrong result is deterministic —
1087631800 reproduced bit-identically across two sessions (cycles 47,954 / 47,952).

This is the **sharpest reproduction of the gp-captable miscompute so far**, better
than `gp_diag3`:

- The delta is minimal and fully enumerated (5 instructions of instrumentation).
- Both variants are tiny and freely modifiable — `gp_diag3` cannot be perturbed
  without the fault vanishing (v4 is clean).
- It is deterministic on both sides.

**Immediate bisection available** (2–3 boots): the instrumentation is three separable
pieces — (a) the two stores to `res[65]`, (b) the two `csrr minstret`, (c) the store
to `res[64]`. Build a variant with each alone and find which one triggers it. That
localizes the trigger to a single construct, which no amount of gp_diag probing
achieved.

Note the instrumented variant is only **+172 cycles** over the control, so whatever
goes wrong is not a wildly different execution path — it computes a wrong checksum in
nearly the same time.

## 2. Reproducibility datum (the paper needed one)

The un-instrumented `beebs_prime` has now been measured in two independent sessions
a day apart, with a full power-cycle and firmware reload between:

| session | cycles |
|---|---:|
| 25-07 | 47,804 |
| 26-07 | 47,780 |
| | **−24 (−0.05%)** |

`ref/fpga-silicon-measurements-for-paper.md` listed "no error bars" as a limitation. This is a start: the
measurement is reproducible to **0.05%** across sessions on this rung. (Still one
repeat, not a distribution.)

## 3. coremark_matrix HANGS — it was never transfer-blocked

```
transfer /tmp/coremark_matrix.dom: 2316 b64 chars      <- transferred cleanly
coremark_matrix: no END marker in 120s (attempt 1)
coremark_matrix: no END marker in 120s (attempt 2)
```

The dom transferred without error and the `cscall` then produced nothing, twice.
**Finding #2 of the 25-07 note (the `-Os` domain-entry hang) is confirmed on a fresh
dom**, and that note's "transfer never landed" was the inverted causation predicted in
`26-07-2026_16-41-09_tier2b-feasibility-...md`: the hang wedges the console, and the
*next* transfer fails.

⇒ Task #46's premise is settled. It also retires the last non-SQLite justification
for tier-2b: a bigger delivery channel would have changed nothing here.

## 4. FPGA-launch fixes made this session

**The console had never used websockets.** `python-socketio[client]` is supposed to
pull `websocket-client`; it was not installed, so every board session to date ran
over **HTTP long-polling**. That is a plausible common cause of the `fast_put`
wedges, the `BadNamespaceError` socket drops, and the transfer slowness that has cost
several boots. System pip is PEP-668 locked, so it lives in a venv:

```sh
python3 -m venv --system-site-packages /tmp/capstone-b/fpga-venv
/tmp/capstone-b/fpga-venv/bin/pip install websocket-client
# run the driver with /tmp/capstone-b/fpga-venv/bin/python
```

Nothing system-wide was modified. The control run above used it and the
"only polling transport is available" warning is gone. **Whether it actually lowers
the flake rate is not yet established** — one clean run is not evidence. Judge it
over the next several sessions.

**Boot+transfer is now a retryable unit in the perf runner.** The control was lost in
the first session to a GDB timeout inside `cold_boot` (`monitor reset halt`), and the
runner *skipped the rung* — silently dropping the one measurement that session
existed to make. `cold_boot` is idempotent, so a retry costs a reload; losing a rung
costs the conclusion. (`run_ladder_base_fpga.py` already had this; the perf runner
did not.)

## 5. Unvalidated lead: the 4 KiB code window may not be real

`link-gpfree.ld` forces globals to image offset `0x1000`, so every domain's `.text`
must fit **4096 bytes** — which is why the rungs are near-microbenchmarks and why
full CoreMark/Dhrystone cannot be built at all.

But the constraint looks like one hardcoded number, not a hardware limit:

- the monitor splits at a **runtime** `code_size` the controller passes
  (`__split(dom_code, base_addr + code_size)`), and the controller passes the
  **whole image size** — so PCC already covers ~9.5 KB, not 4 KiB;
- `gp` is derived from `dom_data` by carving 16-byte slots off its **end**
  (`gp_captable_build.inc`); it never references `base+0x1000`;
- `GPFREE_GLOBALS_OFFSET` appears **only in the linker script's comments** — no code
  reads it.

`coremark_matrix` rebuilt against a 16 KiB-window variant **links cleanly**. That is
all that has been shown. Residual risk is the large-RO delivery path for
**initialized** globals (`__gpfree_globals_base` offsets) — `coremark_matrix` has 1
global, 0 initialized, so it does not exercise it. **Next: QEMU-validate with a rung
that has initialized globals**, then silicon. If it holds, this is the highest-leverage
fix for benchmark representativeness — bigger than any selection decision.

(Build knobs added for this: `LINKER_SCRIPT` and `DOMAIN_EXTRA_CFLAGS` in
`build-ladder-domain.sh`.)
