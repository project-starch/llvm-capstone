# OPEN: gp-captable domains miscompute a global-array store+accumulate loop on silicon (correct on QEMU)

> ## BOARD RESULT 2026-09-05 — DOES NOT REPRODUCE on the resident bitstream. Not yet attributed.
>
> B4 of the cycle-2 regression sweep, control-valid boot (`k800` retval 4), firmware
> `4a5677c73b0eedd8`, bitstream `caplifive_s12fix_5097eb166.bit`, cycle-2 compiler, `-O0`
> silicon config, positions 2 and 3 ahead of the risky arm:
>
> ```
> rc_const0   2016   (oracle 2016)   the matched control, as in July
> rc_p1       2080   (oracle 2080)   the July FAILING arm — now returns the correct value
> ```
>
> **The July signature — an address-like value unless the stored value equals the loop index —
> did not appear.** `rc_p1` returned exactly the native answer.
>
> **What this is evidence FOR: the bug is not live in this configuration.** What it is NOT evidence
> for, and must not be quoted as: "the bug is fixed", or "the S-12 fix fixed it". Four things
> changed between July and this run and this single reading cannot separate them:
>
> 1. the bitstream — `s12fix` carries the S-12 forwarding fix and **lacks** the R-20 fix
>    (`2efb3604f`, verified by `merge-base`);
> 2. the compiler — cycle 2, though the `rc` pair's codegen was diffed against the July claim and
>    is the same shape (one `addiw`, same store/reload pair, same registers);
> 3. the reproducer — a **reconstruction** from this document's description; the July `.c` is
>    gone, so "same test" is by design rather than by bytes;
> 4. the R-20 path itself — the rebuilt R-20 repro read `0xD0000000` in B1 of the same session,
>    which points the same way but was pre-ruled inconclusive on a rebuilt draw.
>
> **The deciding reading is B6:** the FROZEN `sbx8.dom`, byte-exact, on this bitstream, with the
> control relinked to `0x20000` to avoid the R-3 collision. If the frozen R-20 repro also reads
> `0xD0000000`, the `s12fix` lineage cures the x10 forwarding path by a route other than
> `2efb3604f`, and R-20 — whose signature is *exactly* this bug's — is the probable cause of what
> was measured in July. If it reads `0xD0000001`, R-20 is live and this bug was something else
> that has since gone.
>
> Either way the status moves from OPEN to **NOT REPRODUCED — attribution pending B6**, and the
> blocks it carried (silicon-compatibility claim, branch merge, app-level silicon perf) are no
> longer supported by a live failure. They should not be lifted on this reading alone; they
> should be lifted when B6 says which fix did it.


> ## UPDATE 2026-09-05 — this may already be FIXED, and the test is cheap
>
> This document's own conclusion is that the signature is **"a microarchitectural store/forwarding
> race in the RTL LSU, not a deterministic compiler bug"** — non-deterministic address-valued
> garbage (`0x8B8x…`), well-formed capabilities, codegen provably equivalent between the passing
> and failing arms, QEMU always correct.
>
> **S-12 is a store/forwarding defect in exactly that path, and it was root-caused, fixed in RTL
> and flashed on 2026-09-04** — six weeks after these measurements. A capability store's
> scoreboard `rd` is aliased to its own store-data register; under store-buffer back-pressure the
> commit stage holds `we_gpr` while withholding `commit_ack`, the WAW guard clears, and forwarding
> hands a younger consumer the wrong value. "An integer accumulator receiving a capability cursor,
> only in a loop that also stores, only on silicon, non-deterministically" is that shape.
>
> **This was not connected at the time because S-12 had no mechanism until 2026-09-03.** The
> connection was suggested by the compiler lane on 2026-09-04 from the signature alone.
>
> ### CORRECTION, same day: R-20 fits the symptom BETTER than S-12, and it is a confound
>
> **R-20 is "a capability store loses its x10 clobber claim, so a later reader of x10 gets the
> store's BASE ADDRESS instead of the loaded value"** (`ref/ISSUES.md` under S-03). That is not
> merely S-12-adjacent — *"gets the store's base address"* **is** the address-like value recorded
> here. S-12's forwarding delivers `cnull` (cursor 0), which does not match this symptom; R-20
> delivers an address, which does.
>
> **And the R-20 COMPILER workaround landed `30c275b5d781`, 2026-08-10 — eighteen days AFTER these
> measurements.** So the July binaries did not have it and any rebuild does. That is a confound in
> the opposite direction from the one noted below: a rebuilt `rc_p1` that passes may be passing
> because of a *compiler* change from August, not the RTL fix from September, and the bug may have
> been silently closed for a month.
>
> The compiler lane's objection to the S-12 framing was correct and is what led here: a forwarding
> path returning a register's stale contents predicts the reload reading `i`, not an address.
>
> ### SECOND CORRECTION, same night: there is NO 2x2 — the workaround was already reverted
>
> **The R-20 compiler workaround was reverted on 2026-08-10 at 19:39, FOUR HOURS after it landed
> at 15:22 the same day** (`cdbb92360e2b`, *"the silicon fix makes it unnecessary"*). It is an
> ancestor of `dev`, and the revert's own post-check was that `git diff 30c275b5d781^ -- llvm/` is
> **empty**. So it is not in today's compiler, there is nothing to revert, and the claim below —
> that a rebuild carries the workaround while July's binaries did not — is **false**. Both lack it.
>
> **The confound I asserted does not exist.** Struck rather than deleted, because the reasoning
> that produced it (workaround dated after the measurement → present in any rebuild) is exactly
> what a later reader will re-derive from the commit dates alone. The date was right; the lifetime
> was four hours and nobody checks a lifetime.
>
> ### What the test actually is: ONE arm, and it is cleaner than the 2x2
>
> **R-20 was fixed IN SILICON around 2026-08-10.** The revert commit carries the evidence:
> `caplifive_r20.bit` holds the RTL fix, the package's own 13 KB repro goes
> `0xD0000001 → 0xD0000000`, and the SQLite-level site returns where it used to wedge. These
> measurements were taken **2026-07-23, on a bitstream without that fix.**
>
> So: run `rc_p1` (with `rc_const0` as the matched control) on the resident bitstream. If it
> returns 28 rather than an address, **this bug has been fixed in hardware for about a month and
> nobody noticed** — and the candidates are the R-20 RTL fix, whose symptom is *literally* "a later
> reader gets the store's base address", or the S-12 fix.
>
> **Establish the lineage first or the result is uninterpretable:** does
> `caplifive_s12fix_5097eb166.bit` descend from `caplifive_r20.bit`? If NOT, a pass points at S-12
> and a fail says nothing about R-20. If it does, a pass is consistent with either and a fail
> refutes both. **Not established here.**
>
> ### ~~The experiment this actually needs — a 2x2, not a single arm~~ (WITHDRAWN, see above)
>
> The workaround commit says **"TEMPORARY. Revert when a bitstream carrying the RTL fix is
> resident"**, with revert instructions in
> `../../tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/WORKAROUND.md`. So both arms are
> buildable and the hypotheses separate:
>
> | | workaround ON (today's default) | workaround REVERTED |
> |---|---|---|
> | `rc_p1` passes | the fix is in *one* of them — undetermined | **the resident bitstream fixes R-20 in hardware**, and the workaround can be retired |
> | `rc_p1` fails | neither fixes it; the July bug is still live and is something else | the workaround is what carries it; keep it |
>
> The reverted arm is the informative one, and it answers a standing question worth money on its
> own: **can the R-20 workaround be retired?** It costs codegen quality on every capability store,
> and nobody has been able to test the condition its own commit names.
>
> **Whether the resident `caplifive_s12fix_5097eb166.bit` carries the R-20 RTL fix (branch
> `r20-fix`, `2efb3604f`) is NOT established here.** Establish it from the bitstream's lineage
> before reading the 2x2, or one row is uninterpretable.

> ### The test, and the confound that has to be controlled
>
> Rebuild `rc_const0` (`acc[i]=i; s+=acc[i]` — PASS) and `rc_p1` (`acc[i]=i+1` — FAIL) and run both
> on `caplifive_s12fix_5097eb166.bit`. If `rc_p1` now returns 28 instead of an address, this is
> closed and with it the block on the silicon-compatibility claim, the branch merge, and app-level
> silicon perf.
>
> **The confound: the reproducers no longer exist.** No `.c`, no `.dom`, no build command survives
> — only the table in this document. So a rebuild uses **today's compiler**, not July's, and a pass
> would be consistent with *either* the RTL fix *or* a codegen change. To attribute it, the
> disassembly of the rebuilt pair must be checked to still be the same shape (store-and-accumulate
> in one body, the same capability-typed operands); the compiler lane has offered exactly that
> diff. Without it, a green result is suggestive and not conclusive.
>
> This is the third measurement in this repository found on 2026-09-04/05 to be unreproducible from
> its own record — see `../../tests/fpga-repros/S13-o1-dyn-rev-node-hang/` and the note in
> `../../tests/fpga-repros/README.md`. The loops here are three lines and rebuildable, which is the
> only reason this one is recoverable at all.


> ## ⚠️ 25-07-2026 — THE ROOT CAUSE BELOW IS REFUTED. READ THIS FIRST.
>
> Everything below concluded the cause is an **RTL `shrink`→store forwarding hazard**, with
> "build with shrink off" as a proven workaround. **That does not hold.** The 4 silicon-ladder
> rungs that miscompute are built shrink-off and contain **zero `shrink` instructions**, and
> still fail on hardware. Also refuted since: bounds-representability (the rung with the
> *largest* global passes), "array store with a live accumulator" (`beebs_recursion` has no
> array), and any instruction-level discriminator.
>
> **The mechanism is currently UNKNOWN. Do NOT escalate the shrink story to the board owner.**
> Full evidence + the `gp_diag` diagnostic rung built to settle it:
> `history/25-07-2026_17-09-01_gp-captable-miscompute-shrink-theory-refuted.md`.
> The *observations* below (probe values, board discriminators) remain valid data; only the
> interpretation is withdrawn.

**Status: OPEN. Our side VALIDATED (caps well-formed on HW, codegen provably
equivalent); now strongly RTL-store-path-leaning — ready to escalate to the board
owner, but not yet confirmed by RTL/gdb.** Do not claim silicon compatibility, do
not merge `capstone-gp-free` until the board owner confirms and/or a workaround is
proven.

## UPDATE 23-07 #3 — shrink RESULT is correct; it's a shrink→store FORWARDING hazard (RTL)

On-board shrink-bounds probe (`shrink_probe.c`: shrink acc cap to 4 bytes, `lcc`
the result). Board == QEMU exactly: **end−base=4, valid=1, base−cursor=0.** The
shrunk cap is valid and precisely bounded on hardware → **NOT representability,
NOT an invalid cap, NOT our codegen making a bad cap.** The earlier representability
hypothesis (#2) is REFUTED.

Refined root cause: the RTL mishandles a **store through a freshly-`shrink`ed
capability** — a `shrink`→dependent-store RAW forwarding hazard. Fits everything:
no-shrink removes the shrink→store dependency (passes); fence-immune (register
forwarding, not memory ordering); non-deterministic (pipeline race); passes only
when value==index (register schedule dodges the hazard). Our side is fully
validated (all caps correct on board, codegen equivalent to the passing case,
QEMU correct). **Confidence ~90% RTL; escalation to the board owner is justified.**

Two workarounds: (1) shrink off (coarser bounds, proven on silicon); (2) likely,
break the shrink→store adjacency (schedule an instruction between them) to keep
per-element bounding — untested, worth trying in codegen.

## UPDATE 23-07 #2 — LOCALIZED TO `shrink`; WORKAROUND FOUND

Board discriminators (run8/run9, fast transfer):
- **Fence does NOT fix it** (`fence rw,rw` between cap store and accumulate, both
  reload and register-sum forms still garbage) → not a memory-ordering hazard.
- **Stack arrays fail too** (`stack_array`, sp-derived cap → 0x0D1FC5A8) → NOT
  gp[0]-specific; any capability store to an array element in a loop.
- **NO-SHRINK PASSES** (the decisive one): `rc_p1` and `stack_array` rebuilt with
  `-capstone-shrink-stack=false -capstone-shrink-globals=false` (per-access
  element-`shrink` removed) return the CORRECT 1000036 on the board. With shrink
  on they garble.

=> The fault is the per-access **`shrink`** instruction (element-bounding), not the
store/load/gp path. **Workaround: build domains with shrink off** — array accesses
fall back to whole-array/frame bounds (coarser but still capability-bounded);
real programs then run correctly on silicon (unblocks app-level silicon perf).

**Likely mechanism (hypothesis, connects to a known issue):** shrinking a cap to a
4-byte element may produce bounds that are **unrepresentable under the RTL's
capability compression** (cf. the cursor-0-unrepresentable issue,
[[project_silicon_gp_delivery_boardowner_guidance]]). Our QEMU fork likely models
precise/uncompressed caps and so never sees it; the RTL enforces compression and a
mis-rounded bound then makes a subsequent store land wrong (address leaks into the
result, non-deterministic by data-region placement). If so this is arguably OURS to
fix (make per-access shrink representability-aware) rather than a pure RTL defect —
NOT yet confirmed. Decisive next probe: `lcc` the shrink'd cap's base/end on the
board vs QEMU; a mismatch = representability (ours), correct bounds + corrupt store
= RTL. Store-value dependence (passes only when value==index) is still unexplained
and fits a compression/rounding interaction.

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
