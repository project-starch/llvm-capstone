# RESULTS: three benchmarks measured on silicon; two blocked by a domain-entry hang

**Date:** 2026-07-26 · **Lane:** B · ~14 board boots. Board powered off + unlocked.

## The result

Each kernel compiled twice from the identical source header by the **same clang at
the same `-O`** — once as a pure-capability domain, once for plain `riscv64` — and
run on the same board. Baseline is the **warm** pass (the capability domain has no
paging). Both halves bracket the compute only.

| benchmark | opt | capability | baseline | **cycles** | cap instret | base instret | **instr** |
|---|---|---:|---:|---:|---:|---:|---:|
| `beebs_prime` (pure scalar) | −O0 | 47,780 | 46,306 | **1.032×** | — | 14,680 | — |
| `rv8_primes` (sieve, 16.5M cyc) | −O0 | 17,283,292 | 16,459,057 | **1.050×** | 8,773,753 | 7,960,829 | **1.102×** |
| `beebs_recursion` (deep + mutual recursion) | −O1 | 18,957 | 10,523 | **1.801×** | 2,944 | 2,019 | **1.458×** |

**Pervasive spatial safety costs 3.2% on scalar code, 5.0% on array code, and 80% on
deeply recursive code.** That spread is the result — a single averaged number would
have hidden the mechanism.

Why recursion is so much worse is visible in the counters: it retires 45.8% more
instructions (vs 10.2% for the sieve) *and* its CPI rises from 5.21 to 6.44. A
gp-free call/return plus capability spills to the stack are paid per call, and
`beebs_recursion` is nothing but calls. The sieve amortises its `ldc` cap-table
indirections over long straight-line loops; recursion cannot.

Both `beebs_recursion` measurements are **certified clean**: its two baseline passes
retired byte-identical instruction counts (2,019/2,019), so no page fault or
interrupt was counted in either, and the capability run reported a clean `phase=2`.

**Caveat on uniformity:** each pair is internally consistent (same compiler, same
level, both sides), but the *set* mixes levels — two rungs at −O0, one at −O1.
`beebs_recursion` is measured at −O1 because that is the level at which it computes
correctly on silicon (below).

## −O1 clears the −O0 miscompute

`beebs_recursion` returned 2095861164 at −O0 against oracle 1579141629. At −O1 it
returns **1579141629**, correct, with a clean phase slot — so at −O1 even the extra
region stores that trigger the −O0 miscompute do not.

That follows the bisect: at −O0 every region store is preceded by an `ldc` reload of
the region capability, and −O1+ keeps it in a register. Optimisation level is a real
workaround for the miscompute, not a guess.

## The remaining blocker: a domain-entry hang, and three things it is NOT

`matmult_int` and `coremark_matrix` do not produce a result at any reachable
configuration. They transfer cleanly, then the `cscall` yields no END marker.

| rung | config | outcome |
|---|---|---|
| matmult_int | −O0 | miscompute |
| matmult_int | −O1 | **hang** |
| matmult_int | −O2 | **hang** |
| coremark_matrix | −Os (4 KiB window) | **hang** |
| coremark_matrix | −O0 (32 KiB window) | **hang** |

Three hypotheses killed:

1. **Not `-Os` codegen.** Finding #2 of the 25-07 note framed this as an `-Os`
   problem. `coremark_matrix` hangs at **−O0** too. That framing is retracted.
2. **Not code size.** `coremark_matrix -Os` hangs with 1,988 B of text — *smaller*
   than `beebs_prime` (1,908) and `rv8_primes`, which pass.
3. **Not any instruction.** Comparing mnemonic sets across all four hanging builds
   and all three passing ones: there is **no** instruction present in every hanging
   build and absent from every passing one, nor the converse.

Also killed, for the third time and recorded so it is not resurrected: **variable
`cincoffset` does not discriminate.** It looked compelling (`matmult_int -O1` uses
13, `beebs_recursion -O1` uses none) until `beebs_prime_noins` (PASSES) and
`beebs_prime_m3` (FAILS) turned out to have an identical count of 7.

⇒ The hang is not reachable from the compiler side. It is a domain-entry fault
needing monitor/RTL work; the domain-boundary `fence.i` patch
(`patches/opensbi-capstone-sbi-domcall-boundary-fence-i.patch`) is the standing
candidate and has never been built into board firmware.

## The 4 KiB code window is raisable — and it mattered

`link-gpfree.ld` forced `.text` into 4096 bytes. The limit is one hardcoded number,
not a hardware constraint: the monitor splits at a **runtime** `code_size` the
controller passes (the whole image size), `gp` is carved from `dom_data`'s end, and
`GPFREE_GLOBALS_OFFSET` appears only in comments.

QEMU-validated at **16 KiB** with `beebs_insertsort` — chosen because it has an
**initialized** global and so exercises the large-RO delivery path that was the
residual risk — and at **32 KiB** with `coremark_matrix -O0`.

This stopped being theoretical: `coremark_matrix` needs 17,955 B of text at −O2, so
it fits the 4 KiB window **only** at −Os, the level that hangs. At 32 KiB it builds
at −O0 (5,316 B) and −O1 (6,360 B). The window is no longer what blocks it — the
hang is.

## Build-side limits found while sweeping opt levels

Both on rungs already recommended for dropping from the paper set:

- `beebs_crc32` cannot build at −O1+: the constant-folded 2048 B `.L.crctable`
  overflows the 12-bit store offset in `gen-gp-captable-glue`.
- `beebs_insertsort` **crashes clang** at −O1.

## FPGA-launch fixes

- **The console had never used websockets.** `python-socketio[client]` is supposed to
  pull `websocket-client`; it was missing, so every board session to date ran over
  HTTP long-polling. Now in a venv (`/tmp/capstone-b/fpga-venv`, `--system-site-packages`);
  nothing system-wide changed. Sessions since have been clean, but that is a handful
  of runs, not evidence.
- **Boot+transfer is a retryable unit** in both runners. A GDB timeout inside
  `cold_boot` used to skip the rung and silently drop a measurement.
- `build-ladder-base-fpga.sh` now emits the native oracles it needs; a board run from
  a fresh OUT_DIR previously aborted at the artifact check.
- New build knobs: `LADDER_OPT`, `LINKER_SCRIPT`, `DOMAIN_EXTRA_CFLAGS`.
