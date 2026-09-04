# FPGA gp blocker: full root cause + the plain-call/ret fix (gp-free, cjalr-free domain runs on QEMU)

**Date:** 2026-07-20
**Status:** RESOLVED at the compiler/ABI level; QEMU-validated; silicon run pending (see "What remains").
**Artifacts:** `capstone/tests/rtl-smoke/{start-fpga-nogp.S, borrow_cost_fpga_nogp.c, borrow_cost_fpga_nogp_ctl.c, build-borrow-cost-fpga-nogp.sh}`
(commits `e55e6c5`, `aada422`).

## The one-line result

A **real, globals-using** borrow-cost measurement now runs end-to-end with **no
`gp` and no `cjalr`** — the two constructs that stall on the CVA6/Capstone RTL —
and reproduces the reference numbers under QEMU: **raw=2, borrow=6 cycles/op**
(borrow-at-free = +4, O(1)); copy@256B=227, copy@1024B=899. This is the first
domain in our lineage that is actually silicon-shaped: every prior benchmark
(rv8, coremark, beebs, sqlite) only ever "worked" because QEMU **fabricates** the
`gp` those domains depend on — none ever ran on hardware.

## Why the old domains never ran on silicon (the two layers)

### Layer 1 — the `gp` fabrication (already known, now precisely located)

Our LLVM backend reaches every module global and forms every function code
capability via `cincoffset X, gp, <abs>`, assuming `gp = PCC(cursor 0)`. That
form is not representable/derivable on real 128-bit caps. QEMU makes the domains
"work" by fabricating it in three places (the user's May-19 patches
`7aca0540` / `39130dc1`), confirmed in `capstone-qemu/target/riscv/op_helper.c`:

- `helper_cscincoffset` / `...imm`: `if(!rs1.tag && priv==PRV_C && rs1==3) gp = PCC(cursor 0)`.
- `helper_cjalr_switch_caps`: on every cap call in `PRV_C`, `gp = PCC(cursor 0)`.
- `helper_cscall`: same, on domain entry.

On the RTL none of this happens, so `gp` arrives 0 and `delin gp` / `cincoffset …, gp`
stalls (no trap; in-domain faults route to `ctvec`).

### Layer 2 — the capability RETURN (the deeper wall, found today)

Even after removing all globals (so the compiler emits no `cincoffset gp`), our
clang still lowers a C function's **return** to `cjalr zero, 0(ra)` — a
*capability* return that requires `ra` to hold a code capability. The entry glue
must therefore call `domain_main` such that `ra` becomes a code capability, i.e.
via `cjalr ra, 0(<codecap>)`. Forming that `<codecap>` needs `cincoffset gp, …`
(stock `start.S`) — the very `gp` we removed.

**Is there a gp-free way to form a code capability on a `cscall` domain entry?
No.** Verified against the QEMU model (which mirrors the ISA):

- `auipc` yields a **scalar** (pc+imm), not a PCC-derived cap (unlike CHERI `auipcc`)
  — `trans_rvi.c.inc:trans_auipc`.
- There is **no "read PCC into a GPR"** instruction in the whole Capstone set
  (`trans_capstone.c.inc`).
- **CTVEC** is a code cap covering the image — but only *after* it is fabricated
  from `pc_cap` on an **mret to non-M** (`op_helper.c:357`). A `cscall` domain
  entry goes through `swap_c_effective_regs`, which swaps in the domain's *saved*
  CTVEC — an **untagged scalar** for a fresh domain. (Confirmed empirically: `scc`
  on the value read from CTVEC asserts `rs1.tag`.)
- On `cscall` entry the only tagged caps present are `ra` = a **sealed-return cap
  back to the monitor** (`helper_cscall` sets it) and the delivered **region data
  cap** — neither is a code cap over the image; PCC itself is unreadable.

So the code-capability the cap-return ABI needs cannot be materialized gp-free on
entry. This is an **ABI-level** incompatibility, not a missing asm trick.

## The fix — the reference monitor's plain call/ret-within-PCC ABI

The reference OpenSBI Capstone monitor (`sbi_capstone.S`) calls its own routines
with **plain `call handle_exception` / `ret`** — plain `jal`/`jalr` that stay
inside PCC and never touch `gp` or `cjalr`. Plain jumps are bounds-checked
against PCC on fetch, so a call+return that stays in the code image is legal in
c-effective mode. We adopt exactly that for the domain:

1. **Eliminate `gp`** — make the domain global-free (no module statics), so the
   compiler emits no `cincoffset gp`. Achieved by the single-region protocol
   (below), which removes all cross-entry state.
2. **Enter `domain_main` with a plain `call`** (scalar `ra`) in the entry glue —
   no code-cap formation.
3. **Retarget the one capability return to a plain `ret`.** `domain_main` is a
   leaf here (all `measure_*` inline at -O2), so `cjalr zero, 0(ra)` is its *only*
   `cjalr`. `build-borrow-cost-fpga-nogp.sh` compiles to `.s`, asserts exactly one
   such return, rewrites it to `jalr zero, 0(ra)`, and asserts the result is both
   **gp-free and cjalr-free** before assembling+linking.

Plain `call` + plain `ret` both stay within PCC — identical in spirit to the
monitor's own ABI. No `gp`, no `cjalr`, no code-capability formation anywhere.

(The "right" long-term fix is a backend option to lower intra-domain calls/returns
to plain `jal`/`ret` for the Capstone target, instead of a post-compile `.s`
rewrite. The rewrite is a scoped, asserted stand-in that unblocks the measurement.)

## The single-region protocol (how `gp` was removed without losing the measurement)

The old `borrow_cost_fpga.dom` used module statics (`regions[]`, `raw_src[]`,
`copy_dst[]`, `sink`) to carry state across two REGION_SHARE entries and a CALL —
that is the cross-entry state that *needs* globals. The gp-free redesign collapses
everything into **one** `REV_SHARED` region and **one** REGION_SHARE entry:

- The controller (`borrow_cost_fpga_nogp_ctl.c`) creates one `REV_SHARED` region
  (host retains its mapping), maps it, and shares it **once**. That single
  `__domcallsaves(dom, REGION_SHARE, region)` entry *is* the measurement.
- The region is both scratch and results: `[0..63]` = 8 result slots, `[512..]`
  raw_src, `[1536..]` copy_dst. `domain_main(region, func, lin_scratch)` runs all
  measures and writes the slots; the host reads them straight back.
- `measure_borrow` runs on a **linear scratch cap** the entry glue carves off the
  stack top with `split` (revoke cost is provenance-independent — the O(1) claim —
  so a stack-carved linear cap is faithful). The DCE sink is a passed pointer.

No statics ⇒ no `gp`. One entry ⇒ no cross-entry state.

## What remains (silicon run)

QEMU is a functional model (no gp/cjalr *fault* modelling beyond what we exercised,
and it fabricates gp) — so the QEMU pass proves **functional correctness and the
gp-free/cjalr-free shape**, not silicon behaviour. To get the cycle-accurate
number the lead wants, the `.dom` + `.user` must run on the board:

- Get the two nogp binaries onto the board — either baked into a rebuilt
  `fw_payload.bin` rootfs overlay (heavy; overlay wiring still open item #2 in the
  rtl-smoke README) **or** transferred over the UART shell at runtime (base64 into
  `/tmp`, the existing built-in image already gives `/dev/capstone` at boot).
- Drive the board via `fpga_driver/run_rtl_smoke.py` (protocol now **verified**),
  adding a nogp suite entry, then harvest the `RESULT` line.

See also: `plans/ndss-pivot-master-plan.md` §8 (gp blocker), the earlier dated
`*_fpga-domain-call*.md` notes, and `reference_fpga_rtl_platform` memory.
