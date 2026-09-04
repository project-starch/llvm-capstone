# Spike: does Option B (intra-domain revoke) really need the gated `start.S` firmware cycle?

*2026-07-09, A-lane. Firmware/monitor code-reading spike (phase 1 of the Q2
de-risk). Status: **hypothesis strongly supported by code reading; one narrow
codegen unknown remains, to be build-verified.** No code changed.*

## Question

The row3 plan (and `history/08-07-2026_13-01-23_...`) framed the literal
single-domain BORROW-REVOKE "after" (Option B) as blocked behind a **big,
sign-off-gated `start.S`/firmware cycle**: `my_first_domain/start.S` delinearises
`sp`/`gp`, so "a domain has no intra-domain linear authority" and cannot mint an
`MREV`. This spike asks whether that wall is actually the blocker.

## What the code says (findings)

1. **The revoke primitives are clang builtins, not monitor-only.**
   `clang/include/clang/Basic/BuiltinsCapstone.td:188,196` define `cap_mrev` /
   `cap_revoke`; `llvm/include/llvm/IR/IntrinsicsCapstone.td:115,130` define
   `int_capstone_cap_mrev` / `int_capstone_cap_revoke`. So **any** TU compiled with
   our Capstone clang — including domain C — can emit `MREV`/`REVOKE`. The monitor's
   `__mrev`/`__revoke` are just wrappers over these (via `sbi_capstone.h`); nothing
   privileges them to M-mode.

2. **The monitor already delivers *linear* capabilities to domains.**
   `sbi_capstone.c`:
   - `split_out_cap(base, len, 1)` mints a **linear** region cap (`__linear`).
   - `shared_region_annotated(...)` `REV_TRANSFERRED` (line 420) hands the domain a
     linear cap with **no monitor-retained revoke** — i.e. the domain *owns* it
     linearly; `REV_BORROWED` (line 388) shares linear with a retained `__mrev`.
   So a domain can be given a linear arena today, through existing SBI ops.

3. **`delin` in `start.S` only touches `sp` and `gp`** (lines 35, 41) — the initial
   stack/global caps, delinearised so ordinary compiled C can copy them freely
   (linear caps can't be duplicated). **Region caps shared later are *not*
   delinearised.** So the "no intra-domain linear authority" statement is too
   strong: it applies to `sp`/`gp`, not to a monitor-granted linear region.

4. **B (task-003) already proved the fault at instruction level:** a held
   register alias and a memory-resident alias of a revoked cap both fault
   (cause-25 / cause-24); an unrelated cap survives.

## Consequence — the wall is narrower than documented

The mechanism for intra-domain `MREV`→`REVOKE` needs **none** of: a `start.S`
change, repurposing `sp`/`gp`, or the `gp`-small-data-region workaround. It needs a
domain to (a) receive a linear arena from the monitor (exists), (b) call
`__builtin_capstone_cap_mrev` on a sub-cap (builtin exists), (c) `cap_revoke` it and
re-touch a cached alias (B proved it faults). The big gated firmware cycle is **not**
on the critical path for the Option B *mechanism*.

## The one remaining unknown (narrow, codegen — not firmware)

Every *existing* probe is **cross-domain**: the monitor holds the `__mrev` handle
and calls `revoke_region`; the domain receives its region via
`SBI_EXT_CAPSTONE_REGION_QUERY` returning a **base address** it casts to a plain
pointer, and deref goes through **ambient / region-mapping authority** (revoke works
by invalidating the region *mapping*). No probe has domain C hold a capability in a
C variable and deref *through it*.

So the open question is purely codegen: **when domain C does
`p = __builtin_capstone_cap_mrev(arena); … cap_revoke(rev); use *alias;`, does the
Capstone clang route the alias deref through the held/derived capability (so revoke
faults it), or re-materialise an ambient pointer (so it doesn't)?** B's task-003
proved the fault holds at the instruction level; this asks whether *compiled domain
C* preserves that. It is testable with a tiny C probe — no firmware, no SQLite.

## Residual for the *literal* Option B

"`MREV` SQLite's **own internal** column-name heap pointer" additionally needs
SQLite's memsys5 arena (`sqlite_heap`) to be **linear-backed**, so its allocations
are `MREV`-able. That is a separate, contained heap-backing question — *not* the
`start.S` ABI change. The pragmatic Option B (domain copies the buffer into a
monitor-granted linear arena, `MREV`s that, revokes at finalize, single-domain) does
**not** need it.

## Next step (verification)

1. **B (codegen spike, its lane):** confirm domain C compiled with Capstone clang
   can `__builtin_capstone_cap_mrev` + `cap_revoke` and that a deref through the
   held cap faults after revoke; also report whether `MREV` **consumes or retains**
   its linear source (decides one-grant-one-revocation vs many), and whether a
   *linear sub-cap* (SHRINK/SPLIT of the arena) is `MREV`-able. → answers unknown #3.
2. **A:** if B is green, build a minimal single-domain probe (monitor grants a
   linear arena → domain `MREV`s a sub-buffer → `REVOKE` at a lifecycle point →
   cached alias faults), no `start.S` change. That is the Option B mechanism proof;
   the SQLite integration follows.

## Bottom line for the A/B decision

The spike **lowers the Option B cost estimate**: the "gold standard" likely does
*not* require the big gated firmware cycle for its mechanism — only a narrow codegen
confirmation (B) plus a normal probe build (A). This strengthens the case for
heading toward Option B rather than settling for the two-domain Option A, pending
B's codegen result. The `start.S` linear-authority cycle is still what rows 11/14/#78
need for *their* reasons, but it is **not** the row3/13/18/19 Option B gate.

---

## RESOLVED (2026-07-09, B task-005) — mechanism confirmed at C level; recipe fixed

B ran the codegen spike (9/9 green, firmware-free, at
`capstone/capstone-qemu/tests/capstone-mrev-codegen/`; super `68ac184c` /
submodule `fd4bc0c0` on `capstone-bootstrap-b`; investigation
`history/09-07-2026_20-42-10_intra-domain-mrev-codegen-spike.md`). The narrow
unknown is **closed — favourably:**

1. **Deref-through-held-cap after REVOKE → FAULTS.** All three paths fault, on
   *compiler-lowered* C: register-held alias (cause-25), alias passed across the C
   ABI into a non-inlined callee (cause-25), alias spilled to memory and reloaded
   (cause-24). `-O2` asm is direct evidence — the post-revoke load is
   `lbu a1, 8(a1)` off the held cap, **no `auipc`/`gp` re-materialisation** of the
   arena symbol. Codegen does **not** defeat the revoke. → Option B mechanism needs
   no firmware and no codegen fix.
2. **`MREV` RETAINS its source** (`helper_csmrev` copies `rs1`→`rd`, retypes only
   `rd` to REV, never nulls `rs1`). One linear grant mints **many** nested
   revocation handles → arena reuse works.
3. **Linear sub-cap is `MREV`-able via `SPLIT`, not `SHRINK`.** `cssplit` gives the
   new half a fresh rev-tree node at the same depth (mrev+revoke on one half stops
   before its sibling); `SHRINK`/`SHRINKTO` copy `rev_node_id` unchanged, so a
   shrunk sub-cap shares the arena's node and is **not** independently revocable.
   One arena can protect several sub-buffers independently — but only if split.

**The Option B recipe (corrected by B's C3 finding):**
```
arena = <monitor linear grant>;
R     = mrev(arena);          // retains arena (finding 2)
alias = delin(arena);         // REQUIRED: passing a LINEAR cap by value is
                              // silently CONSUMED (movc nulls a non-NONLIN src);
                              // delin only clears the node's linear flag, so the
                              // alias stays revocable and REVOKE returns a reusable LIN.
… use *alias freely …
revoke(R);                    // cached alias now faults
```
Arena must be reachable **only** through the tracked cap (`mrev_ambient_miss`
returns rather than faults — confirms the task-004 provenance rule at C level).

**Three codegen defects surfaced (B reported, did not fix):**
- **C1 — `fastcc` + a capability argument ICEs clang** (`CC_Capstone_FastCC` lacks
  the `MVT::i128` case → `llvm_unreachable` at `CapstoneISelLowering.cpp:23820`).
  GlobalOpt promotes internal-linkage fns to `fastcc` at `-O1+`, so any domain TU
  with a non-inlined `static` fn taking a pointer fails to compile at `-O1/-O2`.
  **Blocks the SQLite integration, not the mechanism.** ~10-line fix; mutates the
  shared LLVM tree A also builds → needs a go.
- **C2 — `cap_mrev` marked pure (`Const/IntrNoMem`) but mutates the rev tree.** At
  `-O2` an unused `MREV` is DCE'd and two `MREV`s CSE into one, though each must
  produce a distinct node. `cap_delin` has the same defect. Correctness bug.
- **C3 — passing a LINEAR cap by value silently consumes it** (only NONLIN is
  copyable; `movc`/`cincoffset` null a non-NONLIN source). Correct linear
  semantics, but C has no model of it and emits no diagnostic. Drives the `delin`
  in the recipe above.

**Method note (B):** a cause-24 "fault" is ambiguous (tag gone = revoked reload
*or* consumed cap); only cause-25 is self-proving (tag intact, node revoked). B's
first draft nearly false-certified on a C3 consume masquerading as a revoke; the
driver now asserts the **exact** cause and every cause-24 expectation carries a
no-revoke control. A's mechanism probe must do the same.

## Revised next step (A)
Build the minimal single-domain Option B mechanism probe using the corrected
recipe, **avoiding the C1 trigger** (inline / no non-inlined cap-arg `static` fns,
or `-O0`) so it does not depend on the C1 fix. C1 is only on the critical path for
the *full SQLite* Option B integration, which is gated behind B fixing C1 (+C2).

---

## Follow-on finding (2026-07-09, A) — the held-cap *delivery ABI already exists*

Scoping the Tier-1/Tier-2 split for the runtime probe surfaced a second
decision-changer, from `sbi_capstone.c` + `my_first_domain/start.S`:

1. **The monitor already hands the domain the real capability, not an address.**
   `shared_region_annotated(...)` ends with
   `d = __domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r);` — where `r` is the
   **linear** region cap for `REV_BORROWED`/`REV_TRANSFERRED` (mrev retained by the
   monitor for BORROWED; not for TRANSFERRED). `call_domain_with_cap` does the same
   with `CAPSTONE_DPI_CALL`. So a **held linear capability is delivered into the
   domain** through the domain-call save/restore ABI.
2. **The domain receives it as a capability.** `start.S` at entry does
   `stc(a1, sp, 80)` (store *capability* a1) and reloads it as `domain_main`'s cap
   argument via `ldc(a0, sp, 80)` (lines 45/81). `a0` is handled as a scalar
   (`sd`/`ld`, 44/82). So the delivered region cap reaches domain C **as a real
   capability argument**, not a reconstructed-from-address ambient pointer.
3. **Existing probes discard it.** Every runtime revoke probe calls
   `SBI_EXT_CAPSTONE_REGION_QUERY` → `(unsigned long *)base.value` and derefs
   through ambient authority. That is why they are cross-domain / monitor-mediated,
   and why B's `mrev_ambient_miss` rule bit: the ambient path is a *second*,
   un-revocable route to the bytes.

**Consequence — the tier framing was too pessimistic.** The *literal* single-domain
held-cap Option B (Tier-2) is **not** blocked on a missing monitor primitive. The
delivery exists; the only gap is **domain-side glue**: bind the delivered
`REGION_SHARE` cap to a domain-C variable and run B's proven mechanism (`mrev` it →
`revoke` → a cached alias derived from it faults) **inside one domain**, on a
**real monitor-granted cap** rather than a `csdebuggencap` hand-mint.

Also: a "monitor-mediated single-domain Tier-1" is **not** a cleaner separate
artifact. `revoke` sweeps the *junior* lineage, not the root — the lender's own
mapping survives (`borrow-revoke-uaf` reads it fine post-revoke). So a lone entity
cannot revoke-and-fault its **own** root cap; you need either a borrow to a second
entity (≈ the existing 2-entity probe, ≈ Option A) **or** a domain that holds the
mrev and revokes its own junior alias (= the held-cap path). The faithful
single-domain proof therefore *is* the held-cap path.

**Caveat (C1):** a domain payload that passes the received cap through non-inlined
`static` fns at `-O1+` will hit C1 (fastcc + cap-arg ICE). Prototype the
domain-side glue at `-O0` to prove the path; the `-O1/-O2` (and SQLite) build waits
on B's task-006 C1 fix. Plan: `plans/sqlite-row3-option-b-held-cap-probe-plan.md`.
