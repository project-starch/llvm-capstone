# The Anvil front end mis-parses a bare relational operator next to `||` — SEAL is a SPEC VIOLATION on silicon

**Date:** 2026-08-20
**Status:** RTL defect CONFIRMED against the spec. Not fixed here. Not yet reported to the
hardware side.
**Found:** incidentally, while tracing the domain-switcher timing failure. Unrelated to timing.

## The compiler defect

Anvil generates wrong logic when a **relational** comparison (`<`, `>`) appears as a bare,
unparenthesised operand of `||`. The comparison swallows the rest of the boolean expression as
its right-hand side, and since `||` yields one bit, the whole thing folds to a constant.

Two instances exist in the tree. **One is dead code. One is live and violates the spec.**

## Instance 1 — LIVE: `SEAL`'s size and alignment check

Source, `core/anvil_build/capstone_flu_unit.anvil:160,167-168`:

```
    let max_size = 64'd1024 >>
    ...
    } else if(size < max_size || ((rs1.metadata.start&64'd15) != 64'd0)){
        call raise_exception(data.trans_id,ex_code::ILLEGAL_OPERAND_VALUE)
```

Generated, `core/capstone_flu_unit.anvil.sv` (declarations at `:454-455` are `logic[0:0]`,
`$1135` is `localparam logic[63:0] = 64'd1024` at `:2002`, `$1134` is `size` at `:2001`):

```systemverilog
:2203  assign thread_1_wire$1336 = thread_1_wire$1334 != thread_1_wire$1335;  // (start & 15) != 0  -- correct alone
:2204  assign thread_1_wire$1337 = thread_1_wire$1135 || thread_1_wire$1336;  // 64'd1024 || X  ==>  CONSTANT 1'b1
:2205  assign thread_1_wire$1338 = thread_1_wire$1134 < thread_1_wire$1337;   // size < 1  ==>  size == 0
```

Consumers, both arms:

```systemverilog
:4041  assign EVENTS1[172].event_current = EVENTS1[171].event_current &&  thread_1_wire$1338;  // raise
:3577  assign EVENTS1[318].event_current = EVENTS1[171].event_current && !thread_1_wire$1338;  // seal
```

The parse Anvil produced is `size < ( max_size || ((start & 15) != 0) )`.

**Effect on silicon: `SEAL` raises `ILLEGAL_OPERAND_VALUE` only when the region size is exactly
0.** Neither the minimum-size rule nor the alignment rule is enforced for any capability of size
1..1023, nor for any nonzero-size capability with a misaligned base.

### This is a SPEC VIOLATION, not a design choice

`capstone-spec/parts/cap-man-insn.adoc:459-462`, the `Illegal operand value (29)` conditions
for `SEAL`:

> - The size of the memory region associated with `x[rs1]` is smaller than
>   `CLENBYTES * {sealed_cap_size_clen}` bytes (i.e. `x[rs1].end - x[rs1].base + 1 < CLENBYTES * {sealed_cap_size_clen}`).
> - `x[rs1].base` is not aligned to `CLENBYTES` bytes.

with `capstone-spec/attributes.adoc:9` → `:sealed_cap_size_clen: 64`, so CLENBYTES(16) x 64 =
**1024 bytes**. The Anvil **source** encodes both rules correctly — `max_size = 64'd1024` and
`(start & 15) != 0`. Only the generated netlist is wrong. The RTL *looks* right on review; the
defect is invisible above the netlist.

(The spec lists a third condition — the region `[base+CLENBYTES, base+2*CLENBYTES)` must contain
a capability — which neither implementation checks. Separate gap, not caused by this bug.)

### QEMU diverges too, for an unrelated reason

`capstone-qemu/target/riscv/op_helper.c:1007-1010`:

```c
    if(cap_size(&rs1_v->val.cap.bounds) < CAP_SEALED_SIZE_MIN ||
       !cap_aligned(&rs1_v->val.cap.bounds, 4)) {
        CAPSTONE_DEBUG_PRINT("Sealing requires an aligned region of sufficient size\n");
    }
```

It computes the condition and then **never calls `riscv_raise_exception`** — the block is
debug-print-only and falls through to seal successfully. Every neighbouring check
(`:997-1005`, type and permissions) does raise. And the constant is wrong independently:
`capstone-qemu/target/riscv/cap.h:35` has `CAP_SEALED_SIZE_MIN (16 * 33)` = 528, where the spec
says 16 x 64 = 1024.

**Net behaviour, and the divergence that matters:**

| region | spec | RTL | QEMU |
|---|---|---|---|
| size 0 | raise | **raise** | seal |
| size 1..1023, or misaligned base | raise | seal | seal |
| size >= 1024, aligned | seal | seal | seal |

So for **size 0** the two implementations disagree: a domain that seals a zero-length capability
runs clean under QEMU and hard-faults on silicon. For sizes 1..1023 and for misaligned bases
both under-enforce, so there is no divergence to catch there — a shared gap, not a difference.

## Instance 2 — DEAD: the domain switcher's index decode

`core/anvil_build/capstone_dom_switcher.anvil:115`:

```
if *cur_idx < 7'd3 || (*cur_idx > 7'd8 && *cur_idx < 7'd57) {
    call process(64'd16, 1'b1, 7'd66)   // 16-byte stride, metadata
} else {
    call process(64'd8,  1'b0, 7'd66)   // 8-byte stride, no metadata
}
```

generates (`core/capstone_dom_switcher.anvil.sv:259-270`, widths at `:236-240` all `logic[0:0]`)
a fully re-associated tree equivalent to
`cur_idx < ( 3 || ( cur_idx > ((8 && cur_idx) < 57) ) )`, i.e. **`thread_0_wire$19 = (cur_idx_q == 0)`**.
Had it executed, indices 1, 2 and 9..56 would have taken the 8-byte arm, and since `process`
also does `cur_base += data_size`, the whole context-record stride would be wrong.

**It never executes.** `is_full` is hard-wired to `1'b0`:

* `core/anvil_build/capstone_unit.anvilh:376` — `create_result_pack_domain_switch(id, ex, is_full, ...)`
  is the only function that sets `dom_switch_is_full` to anything but a literal zero (`:383`);
  the plain `create_result_pack` hard-codes `dom_switch_is_full = 1'b0` at `:367`.
* It has exactly **two** call sites in the whole tree — `capstone_dyn_unit.anvil:267` (CALL) and
  `:302` (RETURN) — and **both pass `1'b0`** as the third positional argument.
* `core/ex_stage.sv:1191`, `core/cva6.sv:1545`, `core/commit_stage.sv:356` are pass-throughs; no
  other RTL drives it.
* So `core/capstone_dom_switcher.anvil.sv:697` — `EVENTS0[52] = EVENTS0[2] && !thread_0_wire$8`
  — is the branch every switch on silicon actually takes, and it uses the **correct** predicate
  `thread_0_wire$160 = cur_idx_q < 7'd3` (`:410-411`, a genuine 7-bit compare).

**This reconciles the contradiction** that made me hold the finding: domain switching works, and
SQLite's hundreds of switches are unaffected, because the broken branch is unreachable. It arms
the moment anyone passes `is_full = 1'b1`.

## Scope of the compiler bug — narrow, and pinned

Controls, all verified in generated SV:

* **Bare `<` with no `||`/`&&` beside it → CORRECT.** `capstone_dom_switcher.anvil:121` →
  `.anvil.sv:410-411`, a true 7-bit `cur_idx_q < 3`.
* **Bare `==` next to `||` → CORRECT.** `capstone_dyn_unit.anvil:62` → `.anvil.sv:1721,1727,1728`.
  So it is not `||` proximity as such; it is relational operators specifically.
* **Individually-parenthesised comparisons around `||` → CORRECT.** `capstone_dyn_unit.anvil:122`
  (SPLIT) → `.anvil.sv:1942,1946,1947`.
* **`perm&3'd6!=3'd6` (`capstone_flu_unit.anvil:165`) → CORRECT.** Anvil binds `&` tighter than
  `!=`, opposite to C but matching the intended reading. I had flagged this as a suspected second
  bug class; **REFUTED**.

Every other compound boolean in the tree wraps each comparison in its own parentheses — the
verified-correct idiom. Those were matched by pattern, **not individually traced**:
UNRESOLVED-by-direct-trace, low risk. Whether `<=` / `>=` share the defect is **untestable** —
no bare instance exists in the tree.

## Recommended actions, not taken here

1. **Fix the source defensively** by parenthesising both comparisons at
   `capstone_flu_unit.anvil:167` and `capstone_dom_switcher.anvil:115`. That corrects the
   netlist without waiting on a compiler fix, and it is a strict improvement either way.
2. **Report the front-end bug** to whoever owns Anvil — a mis-parse that silently produces a
   constant is the worst possible failure mode, because review of the source cannot catch it.
3. **A directed simulation test for SEAL** — seal a 512-byte region and a misaligned region, and
   assert `ILLEGAL_OPERAND_VALUE`. Both currently pass. There is no such test today, which is
   why this survived.
4. **Fix QEMU's helper** to raise, and to use 16*64 rather than 16*33.
5. **Add an elaboration-time check** anywhere a generated predicate is expected to be non-constant.

Item 1 changes silicon behaviour (SEAL starts rejecting regions it currently accepts), so it
must not ride along with an unrelated change and needs its own regression run.

---

# AUDIT UPDATE, same day — mechanism SUPPORTED, but three things above are wrong

An adversarial audit was run before this went anywhere. The mechanism survived every attack; all
eight quoted generated-SV lines re-verified verbatim, and the artifact identity was closed
(`sha256 b594100a051c...` identical across the worktree and **both** synthesis tarballs; the
source line has not changed since `304fb5080`, 2025-05-01, so every bitstream since carries it).

## 1. SEVERITY UP — this is a security bug, not a robustness bug

I had guessed the under-enforcement might be harmless because a later access would be
bounds-checked anyway. **That is refuted.** A sealed-return capability's LDC/STC is bounded
against constants derived from `start` and **never consults `end`** —
`capstone_dyn_unit.anvil:322-324`:

```
    let rs1_end            = rs1.metadata.end - 64'd16          // the NORMAL path uses end
    let rs1_start_sealedret = rs1.metadata.start + 64'd48        // sealed-return: start only
    let rs1_end_sealedret   = rs1.metadata.start + 64'd1008      // sealed-return: start only
```

That check is itself generated correctly (`capstone_dyn_unit.anvil.sv:4032-4033`) — its source
parenthesises its operands. And it *must* work that way, because the spec says so:
`capstone-spec/parts/prog-model.adoc:91` — *"`end` … Not applicable when `type = 4` (sealed) or
`type = 5` (sealed-return)."*

**So SEAL's minimum-size precondition is the only thing standing between a small region and a
1024-byte access window.** Under-enforce SEAL and a 64-byte linear RW capability becomes ~960
bytes of read/write authority the holder never had. That is authority amplification, and it
explains *why* the spec picked 1024: the sealed-return window is exactly `start+48 .. start+1008`.

A second path reaches the same place: `dom_switch_data_req_t` (`core/include/ariane_pkg.sv:219-223`)
carries a bare `logic [63:0] base_addr` with no bounds and no capability, and nothing downstream
re-checks the walk against the sealed capability's extent.

## 2. THE CAUSE IS UNRESOLVED — do not write "the Anvil front end mis-parses"

The sibling line `capstone_flu_unit.anvil:165` (`perm&3'd6!=3'd6`) is a bare relational adjacent
to a bitwise operator and generates **correctly** (`.anvil.sv:2134,2136`). That rules out a simple
flat/right-associative story. The model consistent with every observation is that **Anvil binds
logical and bitwise operators TIGHTER than the comparison operators** — the opposite of C, and
internally self-consistent.

Under that reading the defect is in the **RTL source** — C-precedence assumptions written in a
language that does not have them — not in the compiler. That distinction decides who owns the fix,
so it must not be asserted either way yet. Write "the source relies on C operator precedence,
which Anvil does not use" and cite the netlist as evidence of effect.

Settling it is cheap and blocked only on access: there is no Anvil compiler in-tree
(`which anvil` is empty; the Makefile says `ANVILC ?= anvil`). Either read the upstream precedence
table or run `anvil` on a five-line case.

## 3. "SPEC VIOLATION" is not yet a safe label — the tree holds THREE different minimums

| where | minimum |
|---|---|
| `capstone-spec` and `capstone-academic-spec` (byte-identical over the `[#seal]` block) | **1024 B** (64 x CLENBYTES) |
| `capstone-qemu/target/riscv/cap.h:35` | `16 * 33` = **528 B** |
| `capstone-c/samples/capstone.h:36` | `CAPSTONE_SEALED_REGION_SIZE 36`, used as `runtime->malloc(...)` — **36 B as written** |
| the Anvil source's intent | 1024 B |

The reference emulator and the reference C runtime were built to different conventions. If the RTL
were "fixed" to enforce 1024, `capstone-c`'s own samples would start faulting. **Someone has to say
which number is authoritative before this is labelled a violation of anything.** Also: the RTL
implements only two of the spec's three `Illegal operand value` conditions — the third, that
`[base+CLENBYTES, base+2*CLENBYTES)` must contain a capability, is absent from the Anvil source
entirely.

## 4. My QEMU framing was backwards, and the reachability is stronger than stated

I wrote a table implying a live RTL-vs-QEMU divergence at size 0. Two corrections:

* `size = end - start + 1`, so `size == 0` requires `end == start - 1 (mod 2^64)`. **For any
  capability with `end >= start` the exception is not merely rare — it is unreachable.** Say that
  rather than "raises only at size 0", which reads as though a check survives. None does.
* QEMU's block contains **only** a debug print, where the three preceding checks (`:994`, `:999`,
  `:1004`) all raise. So QEMU does not enforce this at all, and its constant is 528. There is no
  reachable divergence to catch; QEMU has its own separate defect deserving its own entry.

## 5. No current caller can trip it — report it as latent, with the amplification noted

The monitor does execute SEAL (`sbi_capstone.c:902`, `__seal(dom_seal)`), on a region of
`DOMAIN_DATA_SIZE = 16 * 96` = **1536 B** (`sbi_capstone.c:187-188`), growing to 2048 B for a
2 MiB domain — and `sbi_capstone.c:730-732` already says in-source *"The seal region only grows
(2048 B), staying above SEAL's 1024-byte minimum."* Someone knew about the minimum and engineered
around it. Base alignment holds via the granule-aligned `split_size`. **No exploit path from the
monitor.** `capstone-c/samples` is the outlier and is not what runs on our board.

## 6. No test would have caught it, and the only SEAL test has never had a positive control

`verif/tests/custom/capstone/sealing.S` is the only dedicated SEAL test. It seals
`.zero 4096*4` = **16384 bytes** (`:73-75`) and exercises only the accept path. Eleven other tests
issue SEAL as setup for CALL/RETURN. **There is no negative test anywhere.** Exactly the
"directed tests that come back clean without ever creating the triggering condition" pattern.

The experiment that closes items 1-6 at once, board-free, ~14 s in Verilator: three arms in the
style of `sealing.S` — a 64-byte region, a `base+8` misaligned region, and a control 16 KiB
aligned region — asserting `ILLEGAL_OPERAND_VALUE` on the first two and success on the third. It
is simultaneously the missing negative test and the positive control `sealing.S` has never had.

**Until it runs, this claim is verified at the RTL-source and netlist level ONLY, never
empirically.** Nobody has executed SEAL with an undersized region on the board, in simulation, or
in QEMU.

## 7. The domain-switcher collapse gets its OWN issue — and one correction to the audit

Per the one-issue-per-folder rule it must not share a folder with SEAL. But the audit's
consequence analysis for it (that a full switch would transfer 544 B where ~944 B was intended)
describes **the unreachable branch**: `is_full` is hardwired `1'b0`, as established above, so the
`val_n = 7'd66` path never executes. The auditor was not given that finding. The collapse is real;
its consequence is latent, not live.

## 8. Two instrument failures, both of which silently produce zeros

* **`grep -E` with a character class returns nothing on this box.** Reproduced directly:
  `grep -c "SEAL" sealing.S` -> **1**; `grep -cE '(^|[^a-z_])SEAL' sealing.S` -> **0**, same file.
  An entire reachability sweep was void because of this and had to be redone with plain patterns
  against a positive control. This is the local `grep` being ugrep, already noted in the `rtl-sim`
  skill for control bytes — it extends to character classes.
* **`awk '/^enum fu_op/,/^}/' | grep -n` counts comment and blank lines**, giving ordinal 211 =
  `XNOR`. Parsing with python (stripping comments) gives 211 = `SEAL`, which is correct. Trusting
  the first would have refuted a true claim on false evidence.
