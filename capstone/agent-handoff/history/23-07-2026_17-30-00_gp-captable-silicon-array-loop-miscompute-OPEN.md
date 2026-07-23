# OPEN: gp-captable domains miscompute a global-array store+accumulate loop on silicon (correct on QEMU)

**Status: OPEN. Our side VALIDATED (caps well-formed on HW, codegen provably
equivalent); now strongly RTL-store-path-leaning — ready to escalate to the board
owner, but not yet confirmed by RTL/gdb.** Do not claim silicon compatibility, do
not merge `capstone-gp-free` until the board owner confirms and/or a workaround is
proven.

## UPDATE 23-07 (cap-field dump on the board) — rules OUR side out

On-board `lcc` probes of the `acc` capability (`cap_probe.c`, position-independent
metrics; QEMU baselines end−base=32, cursor−base=0, perms=7):

| metric | board | QEMU | |
|--------|------:|-----:|--|
| end−base | 32 | 32 | OK |
| cursor−base | 0 | 0 | OK |
| perms | 7 | 7 | OK |

=> the `acc` capability our glue builds is **well-formed on hardware** (correct
bounds/cursor/perms). The miscompute is NOT malformed caps and NOT our cap
construction. Combined with: codegen is provably equivalent to the passing case
(`rc_const0`), QEMU is always correct, and the garbage is **non-deterministic**
(`rc_p1`: 2339323060 then 2341158068 across runs — both address-valued 0x8B8x) →
the fingerprint of a **microarchitectural store/forwarding race in the RTL LSU**,
not a deterministic compiler bug. Escalation to the board owner is now justified;
final confirmation needs their RTL/gdb (we cannot single-step the domain).

The compiler-built `-capstone-gp-captable` domain from Stage 4 runs structurally
correctly on the captype-fixed CVA6 (create / enter / `ldc gp[i]` / `domreturn`,
no wedge) but **returns a wrong value** for a common loop shape. Earlier notes
called this "+80, an -O0 store-to-load hazard, fixed by -O2" — **that was wrong on
every count** (see "What refuted the first theory"). The real signature is broader
and the value is address-like garbage, not a small offset.

## Reproducer (all `-capstone-gp-captable`, same glue, `link-gpfree.ld`, span
0x1020; each returns `1000000 + s` so the raw internal value is visible; **all are
QEMU-correct**)

| probe | body | true | **silicon** | |
|-------|------|-----:|------------:|--|
| `rc_const0` | `for i: acc[i]=i;    s+=acc[i];` | 28 | 28 | PASS |
| `rc_elem`   | `for i: acc[i]=i+100;` then read `acc[5]` | 105 | 105 | PASS |
| `rc_p1`     | `for i: acc[i]=i+1;  s+=acc[i];` | 36 | 2339323060 (0x8B70A0B4) | **FAIL** |
| `rc_x2`     | `for i: acc[i]=2*i;  s+=acc[i];` | 56 | 380843800 (0x16B8A758) | **FAIL** |
| `rc_main`   | `for i: acc[i]=i+10; s+=acc[i];` | 108 | 2338405628 (0x8B5…) | **FAIL** |
| `rc_regsum` | `for i: v=i+10; acc[i]=v; s+=v;` (sum from REGISTER, no reload) | 108 | 2337488124 (0x8B54A3BC) | **FAIL** |

Also correct: `twoloop` (fill in loop 1, sum in loop 2), `single` (one element),
`noacc2` (pure stack loop). Sources in `capstone/tests/runtime-qemu/gp-free-domain/`
(rc_*.c staged under /tmp/capstone) built with `link-gpfree.ld`, `-O0`,
`-mllvm -capstone-gp-captable`; board driver `/tmp/capstone/board_bisect_gpfree.py`.

## What the data says

- **The `acc[]` array itself is fine.** `rc_elem` stores `acc[0..7]` in a loop and
  reads `acc[5]=105` back correctly. Stores land, persist, read back.
- **A real store+accumulate loop CAN work:** `rc_const0` (`acc[i]=i`) is exactly
  correct. The pass/fail split is *value-and-context dependent*, not total breakage.
- **The wrong value is an ADDRESS.** Three failures land at `0x8B5x_xxxx` — a DRAM
  address inside the domain's data region. The accumulator `s` is being
  contaminated with a **capability cursor (pointer)**, not a wrong integer.
- **The `minimal pair`:** `rc_const0` (PASS) vs `rc_p1` (FAIL). Identical code, the
  only source difference is `acc[i]=i` vs `acc[i]=i+1`. Both QEMU-correct. This is
  the cleanest thing to hand to the board owner.

## What refuted the first ("-O0 store-to-load, fix with -O2") theory

- `rc_regsum` sums from a **register** and never reloads `acc[i]` — still FAILS. So
  it is **not** the redundant `-O0` reload.
- `-O2` only ever "passed" because it **constant-folded the whole loop away** (disasm
  was a single `sw 0x2110c6c`, no `acc`, no `gp[0]`). It proves nothing about
  optimized code that genuinely round-trips a global array. **"Just use -O2" is not
  a fix.**
- `rc_const0` passing kills a simple "same-iteration store→load always breaks" story.

## Why `rc_const0` (and only it) passes — leading hypothesis (unproven)

In `-O0` codegen, `acc[i]=i` uses the **same register** for the stored value and the
index basis (`a2=i`, `slli a4,a2,2`, `sw a2`); every failing case uses a **different**
register for the store value vs. the index (`addiw`/`slli` split across `a0`/`a2`).
This *could* be an RTL LSU forwarding/disambiguation corner keyed on register/tag
when a capability store (`sw` via `cincoffset(ldc gp[0], i*4)`) is in flight
alongside the stack traffic for `s` — but it is equally possible our glue produces a
subtly mis-set `acc`/`gp` capability that only bites this pattern. **Not localized.**
We cannot single-step the domain from our side (the harness detaches gdb to get the
Linux shell), so localizing needs the board owner's RTL/gdb access.

## Off-board analysis (23-07) — narrows toward the STORE side

- `rc_regsum` sums from a **register** (no `acc[i]` reload) and STILL fails →
  the fault is not the read-back; merely doing the capability **store**
  (`sw` via `cincoffset(ldc gp[0], i*4)`) in a loop with a live accumulator
  corrupts `s`. Store-to-load *forwarding* is therefore not the (whole) story.
- `rc_elem` reads `acc[5]` back correctly with no fault → the `acc` capability the
  glue builds is **correctly bounded**; our cap construction looks sound.
- Only codegen difference PASS vs FAIL: `rc_const0` uses the **same** register
  (`a2`) for the stored value and the address-offset source; every failing case
  splits them (`a0` = index basis, `a2` = value). Legal RISC-V either way.
- Net: leans toward an **RTL capability-store-path defect keyed on the register
  schedule / a live concurrent stack store**, but our side is not fully excluded.

## Decisive next probe (build for the next, efficient board session)

An on-board **capability-field dump**: after `ldc gp[0]`, `lcc` the `acc` cap and
return base(3)/cursor(2)/end(4)/perms(5). If the fields are wrong on HW → our glue;
if correct → the store path is the RTL. Pair with a direct store/reload-through-two-
cap-regs test returning the loaded word. (Needs hand asm / inline `lcc` — the C
front end won't emit it.) Run it with the faster transfer
(`history/23-07-2026_18-00-00_faster-board-file-transfer.md`).

## Actions

- **Take the minimal reproducer to the board owner** as a *collaboration to
  localize* (QEMU-correct, board-wrong; `rc_const0` vs `rc_p1`), NOT as a "your RTL
  is buggy" claim. Draft: `/tmp/capstone/boardowner-msg-array-loop-miscompute.md`.
- Off-board next: dump the `acc` and `gp` capability fields (base/end/cursor/perms)
  the glue actually builds, and compare the `-O0` register schedules of `rc_const0`
  vs `rc_p1` in detail; try a glue variant that sets the `acc` cap cursor explicitly
  (`scc`) and re-tests, to rule our side in or out before escalating.
- **Blocks:** any "compiler is silicon-compatible" claim, merging `capstone-gp-free`,
  and app-level silicon perf (incl. SQLite) — a real workload will hit this pattern.

## Reusable gotcha (cost two false "crash" board sessions)

A gp-captable domain **must** link with `link-gpfree.ld` (globals forced to
image+0x1000, `readelf -l` span > 0x1000) or the monitor `create_domain` SPLIT is
degenerate and **hangs** — looks exactly like a domain crash (`BEGIN_x`, no
`MARK_PRE_SHARE`). Default `my_first_domain/link.ld` gives ~0x300 span => hang.

Related: `plans/gp-captable-codegen-plan.md`,
`design/gp-domain-global-abi-decision.md`,
`history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md`.
