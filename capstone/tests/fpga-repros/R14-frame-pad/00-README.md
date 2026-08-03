# R-14 — the hardware faults an architecturally legal capability store

**Status: the evidence points at the RTL, not at the compiler.** Measured 2026-08-03 on
`working-caplifive-captype-fixed.bit`.

This package supersedes `../R14-strline-struct/` as the reproducer to hand over. That one needed
a full 1.5 MB SQLite build per variant and its failing shape had four things wrong with it at
once. This one is **two ~10 KB domains whose source differs by a single number**, and it ships
the frozen binaries.

---

## The reproducer

Two domains, identical source except the size of a **dead** `volatile char pad[]` that is
written once and never read for its value:

| domain | pad | emitted frame | `lui`-based frame addressing | result |
|---|---|---|---|---|
| `k800.dom` | `pad[800]` | 3776 B | 0 sites | **returns 4 — correct** |
| `k1200.dom` | `pad[1200]` | 4576 B | 13 sites | **never returns** |

```c
struct kv { const char *z; const char *y; };
struct kv a[32];
volatile char pad[800];            /* <-- the ONLY difference: 800 vs 1200 */
unsigned i; int ok = 0;
pad[0] = 1;
for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
for (i = 0; i < 4; i++)
  if (a[i].z && a[i].y && len(a[i].z) > 0 && len(a[i].y) > 0) ok++;
return (unsigned)ok + (unsigned)pad[0] - 1u;      /* expect 4 */
```

Full sources in `src/`; canonical copies live in
`capstone/tests/runtime-qemu/silicon-ladder/k{800,1200}_kernel.h`.

The padding changes nothing semantically. It changes only how large the stack frame is, and
therefore whether the compiler can reach frame slots with a 12-bit immediate
(`cincoffsetimm off(s0)`) or must build the address with `lui` + register-form `cincoffset`.

## Why this looks like a hardware defect

**1. The capability at the failing address is well formed.** `bnd2.dom` reads every `lcc` field
at `&a[3].y` — the address the failing store targets — and returns an encoded verdict
(`+1` cursor ≥ start, `+2` cursor+16 ≤ end, `+4` start 16-aligned, `+100 × type`).

    bnd2 -> 107   =  type 1 (cap_type 2 = NONLIN, valid for stores)
                     and 7 = 1+2+4, i.e. ALL THREE CHECKS PASS

`bnds.dom` separately returns `end - cursor + 10`:

    bnds -> 1322  =  1312 bytes of headroom, against a 16-byte capability store

So at the faulting address the capability is NONLIN, its cursor is inside its bounds, the
16-byte store fits comfortably before `end`, and the base is 16-aligned. There is no
architectural reason to fault this access.

Neither probe performs a capability store, so neither can wedge — both always return a number.

**2. The same binary is correct under QEMU.**

    bash capstone/tests/runtime-qemu/silicon-ladder/run-ladder-qemu.sh k1200
    -> __CAPSTONE_LADDER_K1200_PASSED__ (retval = 4)

**3. Every compiler-side explanation was tested on the board and refuted.** Each of these is a
separate ~10 KB domain in `capstone/tests/runtime-qemu/silicon-ladder/`:

| hypothesis | probe | result |
|---|---|---|
| merged string constants | `:143` (same literal 8×); `r14b` built with merging **off** | still fails |
| repeated `ldc` from one cap-table slot | `clp16` — 16 dynamic loads of one slot | passes |
| count of `ldc`-from-gp | `cdif8` — 8 loads from 8 distinct slots | passes |
| capability stores as such | `cst8` — 8 capability stores | passes |
| `ldc`-from-gp **and** a store in one loop | `cgs8` | passes |
| frame size alone | `r14sl`, `r14hl`, `cgpad` — big lui-addressed frames | pass |
| loops | `e3rd` (read unrolled) fails; `e4wr` (store unrolled) passes | not loops |
| non-zero `stc` immediate | `zoff` — every store forced to `imm=0`, verified in disassembly | still fails |

## Running it

The domains are **baked into the buildroot image** — do not ship them over UART (see
`capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md`, §"UART TRANSFER IS RETIRED").

```bash
O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
cp -f images/*.dom "$O/" && cp -f images/*.dom "$T/"
cp -f images/lpc   "$O/" && cp -f images/lpc   "$T/"     # the controller
cd capstone/caplifive-system/sw/buildroot
make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
```

`A=linux-rebuild` **first**: buildroot does not track `overlay/` → cpio, so an OpenSBI-only
relink silently ships the old initramfs. Then, from the board shell (or via
`fpga_driver/run_baked_rungs_fpga.py`):

```
/test-domains/lpc k800  /test-domains/k800.dom      # -> RESULT k800 retval=4
/test-domains/lpc k1200 /test-domains/k1200.dom     # -> no RESULT line; core wedged
/test-domains/lpc bnds  /test-domains/bnds.dom      # -> RESULT bnds retval=1322
/test-domains/lpc bnd2  /test-domains/bnd2.dom      # -> RESULT bnd2 retval=107
```

**Run `k1200` LAST, or alone.** A wedged domain takes the core with it, so anything after it in
the same boot produces no result and is not evidence. Put at most one expected-to-wedge domain
in a boot, and read a session no further than its first failure. (This exact mistake produced a
wrong verdict during the investigation: `e4wr` was recorded as failing when it had merely run
after a wedge; re-tested alone, it passes.)

`images/SHA256SUMS` pins the binaries that produced the results above.

## What is NOT established — please read before acting

Three gaps remain. They are small but each could still move the verdict:

1. **Permissions were never read.** `bnd2` checks type, cursor, bounds and alignment but not
   `lcc` field 5 (`perm`). A capability lacking write permission would be a legitimate reason
   to fault, and would point back at the domain glue rather than the RTL. This is the cheapest
   remaining check and should be done first.
2. **The probes measure a capability materialised for a `volatile` access**, i.e. *a*
   capability to the failing address — not provably the same register the faulting `stc` uses.
   Reading `lcc` fields off the base register immediately before the faulting store would
   close this.
3. **No `mcause`/`mepc` has been read for `k1200` itself.** The `mcause=28` (`OUT_OF_BOUNDS`)
   reading quoted elsewhere in `SILICON-BLOCKER.md` came from a different, SQLite-derived
   domain. The baked driver does not yet perform the debug-mux read that
   `run_sqlite_stages_fpga.py` does on wedge.

Also worth stating plainly: QEMU is known to be permissive relative to the RTL on this project,
so "QEMU passes" alone is not proof the RTL is wrong. It is the *combination* of QEMU passing
and the bounds/type being measurably valid that makes the access look legal.

## References

- `capstone/agent-handoff/ref/SILICON-BLOCKER.md` — full measurement trail, including the
  mechanisms that were proposed and then refuted, and the retractions.
- `capstone/agent-handoff/ref/ISSUES.md` — R-14 entry.
- `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` — board procedure, the baked-image
  workflow, and the driver contract.
- `../R14-strline-struct/` — the earlier, larger reproducer this one supersedes.
