# S-11 — SEAL enforces neither its minimum size nor its base alignment

> **2026-10-01 (evening): FIXED IN RTL on `capstone-ariane` branch `sup-call`, audited; RESIDENT since 2026-10-02 in `caplifive_supcall_36a641e0b.bit` (the R-43 acceptance list passed on it unchanged; this folder's seal-size arms have not been run on silicon yet).**
> The lead delegated the decision ("audit and decide"); the fix is the one this folder asked for, plus one thing the
> audit of it found. (a) Every comparison in `func SEAL` parenthesised on its own, and the `+1` dropped so that
> `size = end - start` with the exclusive end (R-32's off-by-one): a region smaller than 1024 bytes or not 16-byte
> aligned raises `ILLEGAL_OPERAND_VALUE` (29). Measured in simulation (`verif/tests/custom/capstone/sup-sealsize.S`):
> 64 and 1023 bytes trap; 1024 and 2048 seal with every canary word intact; a 1024-byte region starting 8 bytes
> into an aligned block traps (the alignment clause alone), the same at +16 seals. (b) The audit refuted the
> fix's stated guarantee as it stood: a SEALED capability carries no bounds in a register, its effective base is
> its CURSOR, so a 1024-byte region sealed with the cursor advanced to +960 would still put the switch's writes
> past the region. SEAL therefore now sets the sealed cursor to the region's START (the spec says the cursor does
> not apply to a sealed capability; QEMU's CALL uses the region's base) -- measured: that cursor arm reads as the
> control. The cursor is NOT checked, deliberately: the resident monitor seals its interrupt-handler region with
> the cursor at +88, which a cursor check would trap at boot. Every known sealer complies with the size and
> alignment (the monitor's domain seals 1536 B page-aligned, its interrupt-handler seal exactly 1024 B aligned, the
> runtime's contexts 1024 B, every directed test >= 2048 B). QEMU's minimum stays 528 with a different cause;
> silicon is now the stricter side, and QEMU's `CAP_SEALED_SIZE_MIN` should be raised to 1024. Note for the two
> branches carrying `seal-minsize-boundary.S` (`board/r35-directed-repro`, `s12-ldc-rolling-filter`): its 1022-byte
> arm now expects the trap. Registry: R-32 (SEAL's half fixed) and R-48 (the SEALEDRET window, the cursor's other
> half, unverified).


> **2026-10-01: the precedence model is now read from the compiler itself, and two lines below are superseded.**
> The Anvil grammar (`lib/parser.mly:87-91` in the pinned `corank/anvil:cva6` image, compiler commit `288373b5`)
> declares `<` `>` `<=` `>=` at the LOWEST precedence, right-associative, below `&&`/`||` (line 90) and below
> `==`/`!=` (line 91). That is the refined model of the 2026-08-23 history note, confirmed at its source, and
> closes the "not yet confirmed against a precedence table" gap recorded further down. Consequences:
> (1) **QEMU now raises** — the "QEMU never raises" line below is stale: `c128-qemu-merge` raises
> `INSUF_CAP_PERMS` on an undersized or misaligned SEAL since 2026-09-26 (commit `6550f19489`; the parent's
> pinned `674cdab03c` has it), with its minimum still **528** bytes against this RTL's intended 1024, and a
> different cause code from the RTL's `ILLEGAL_OPERAND_VALUE`. (2) The **second instance** of the same fold, the
> domain switcher's full-mode slot sizing (`capstone_dom_switcher.anvil`, the 2026-08-23 note), is **fixed** on
> `capstone-ariane` branch `sup-call` (`d38887426`), confirmed by the switcher's own trace: ids 0-2 at 0/16/32,
> 3-8 at 48..88 by 8, 9-56 at 96..848 by 16, 57-66 at 864..936 by 8. [Superseded the same evening by the top callout: fixed.] The SEAL instance here was at that hour deliberately NOT
> fixed on that branch: the threshold (1024 vs QEMU's 528, plus the `+1` below) is a contract decision.
> (3) What the dead check costs a CALL, measured in simulation (`verif/tests/custom/capstone/sup-sealsize.S` on
> `sup-call`, retirement-trace readings): a **64-byte** seal is accepted (LCC type 4), and the CALL's 8-register
> exchange then writes the caller's medeleg/mip/mie images at **+64/+72/+80**, past the capability's exclusive
> end (the callee reads 0 / 0x80 / 0 where the canary held 0xC0C0…; +88 is untouched), while the callee's own
> medeleg loads the canary pattern from beyond the seal. A 2048-byte control keeps every canary word. (4) A
> deterministic census, `verif/sim/anvil-relational-lint.py` (same branch, with a self-test), flags any relational
> sharing a parenthesis group with `&&`/`||`; on the resident sources it flags exactly `flu:215` and the switcher
> line, nothing else. Supervised CALL's `cssupervise` checks the save area's size and alignment and the seal's
> alignment itself; the seal's SIZE it cannot check (corrected 2026-10-01: a SEALED capability carries no end bound in a
> register -- `ariane_pkg.sv` `decompress_cap_metadata` returns `bound_end = 0` for SEALED/SEALEDRET, that field
> holds reg_id/async -- so the 1024-byte minimum the switch's walks rely on is SEAL's alone to enforce, i.e. this
> very defect; until it was fixed that evening -- the top callout -- the resident monitor's 1536-byte seals and the runtime's 1024-byte seals were the guarantee).

> **2026-09-10: CONFIRMED ON THE RTL BY A DIRECTED TEST.** `verif/tests/custom/capstone/seal-minsize-boundary.S` (`capstone-ariane` `f6ec6c198`), 670 cycles: a **1022-byte** region seals without raising, where the spec requires ≥ 1024. Its negative control — sealing a NON-LINEAR capability, which SEAL rejects on a different branch (`UNEXPECTED_CAP_TYPE`, mcause 27) — is the **only** exception in the whole run, so "no arm raised" cannot be the instrument failing to reach SEAL. Until now this folder rested on reading the generated Verilog; it now rests on the hardware.
>
> **AND THERE IS A SECOND DEFECT UNDERNEATH IT.** The Anvil computes `size = end - start + 1` (`flu:159`) — an INCLUSIVE formula under this RTL's EXCLUSIVE `end` — so even once the folded condition is repaired the check admits a region whose real size is **1023** bytes. **Fixing the fold without fixing the `+1` ships a check that is wrong on its first day.** The test's 1023-byte arm exists for exactly that and becomes the discriminator with no rewrite. Tracked as `R-32` in the registry.
>
> **Also worth knowing before implementing:** QEMU never raises on an undersized SEAL either (`op_helper.c:1280-1283` prints and falls through) and its minimum is **528**, not 1024 — so a fix here makes three implementations disagree three ways until QEMU is brought along.
>
> *(Previous line, superseded: "Not exercised in the 2026-09-05 sweep; status unchanged." Sweep table: `docs/plans/bug-sweep-2026-09.md`; registry: `docs/ref/ISSUES.md`.)*

**This is an instruction-semantics defect in `SEAL`, not a cache or write-buffer issue.** If you
arrived here chasing a capability that lost its tag on reload, a stale tag after a plain store, or
a scrubbed capability reading back live, you want **S-06**, **S-07**, **S-09** or **S-10** instead
— those are all write-buffer / tag-path defects and share nothing with this one. This report is
self-contained; nothing below depends on those.

## The defect

`SEAL` is specified to reject a capability whose region is smaller than 1024 bytes, or whose base
is not 16-byte aligned. **On this RTL it rejects neither.** The check is present in the Anvil
source and is correct there; it is destroyed on the way to SystemVerilog, and the resulting
condition is unreachable for any capability with `end >= start`.

Source — `core/anvil_build/capstone_flu_unit.anvil`, lines 160 and 167:

```
    let max_size = 64'd1024 >>
    ...
    } else if(size < max_size || ((rs1.metadata.start&64'd15) != 64'd0)){
        call raise_exception(data.trans_id,ex_code::ILLEGAL_OPERAND_VALUE)
```

Generated — `core/capstone_flu_unit.anvil.sv`, lines 2203-2205, with `$1337` and `$1338` both
declared `logic[0:0]` at `:454-455`, `$1135` the constant `64'd1024` at `:2002`, and `$1134` the
region size (`end - start + 1`) at `:2001`:

```systemverilog
:2203  assign thread_1_wire$1336 = thread_1_wire$1334 != thread_1_wire$1335;  // (start & 15) != 0   -- correct alone
:2204  assign thread_1_wire$1337 = thread_1_wire$1135 || thread_1_wire$1336;  // 64'd1024 || X  ==>  CONSTANT 1'b1
:2205  assign thread_1_wire$1338 = thread_1_wire$1134 < thread_1_wire$1337;   // size < 1
```

`||` yields one bit, and `64'd1024` is always truthy, so `$1337` folds to a constant and the
entire condition collapses to **`size < 1`**. Since `size = end - start + 1`, that requires
`end == start - 1 (mod 2^64)`. **For any capability with `end >= start` the exception is not rare
— it is unreachable.** No check survives.

Both arms of the collapsed condition are consumed as expected, so the two branches are not
swapped: `:4041` `EVENTS1[172]` raises, `:3577` `EVENTS1[318]` proceeds to seal.

## Why it matters: this is authority amplification, not a robustness nit

A sealed-return capability's load/store window is bounded from `start` alone and **never consults
`end`** — `core/anvil_build/capstone_dyn_unit.anvil`, lines 322-324:

```
    let rs1_end             = rs1.metadata.end - 64'd16          // the NORMAL path uses end
    let rs1_start_sealedret = rs1.metadata.start + 64'd48        // sealed-return: start only
    let rs1_end_sealedret   = rs1.metadata.start + 64'd1008      // sealed-return: start only
```

That check is itself generated correctly, and it *must* work that way, because the specification
says so — `capstone-academic-spec/parts/prog-model.adoc:91`: *"`end` … Not applicable when `type = 4`
(sealed) or `type = 5` (sealed-return)."*

**So SEAL's minimum-size precondition is the only thing standing between a small region and a
1024-byte access window** — which is exactly why the spec picked 1024: the sealed-return window is
`start+48 .. start+1008`. Seal a 64-byte region and its holder gains roughly 960 bytes of
read/write authority it never had.

## What the specification requires

`capstone-academic-spec/parts/cap-man-insn.adoc:484-487`, under `[#seal]`, the `Illegal operand value (29)`
conditions:

> - The size of the memory region associated with `x[rs1]` is smaller than
>   `CLENBYTES * {sealed_cap_size_clen}` bytes.
> - `x[rs1].base` is not aligned to `CLENBYTES` bytes.

With `CLENBYTES = 16` (`parts/prog-model.adoc:71`) and `sealed_cap_size_clen = 64`
(`attributes.adoc:9`, at the spec root), that is **1024 bytes** and **16-byte alignment** — exactly
what the Anvil source encodes. `capstone-academic-spec` is byte-identical over this block.

The RTL implements two of the spec's three conditions; the third — that
`[base+CLENBYTES, base+2*CLENBYTES)` must contain a capability — is absent from the Anvil source
entirely. That is a separate gap, mentioned so the two above are not mistaken for the whole list.

The reported exception code is `mcause 30`, from `core/include/riscv_pkg.sv:354`
(`ILLEGAL_OPERAND_VALUE = 30`) rather than from the Anvil enum, whose own comment warns that a
neighbouring encoder is off by one.

## Evidence

`evidence/seal-minsize-align.S` — three arms, each differing from the control in **exactly one**
property, all three running to completion before anything is reported so one defect cannot hide
the other:

| arm | region | differs from control by | required |
|---|---|---|---|
| 1 control | 16 KiB, 16-byte aligned base | — | must not trap |
| 2 undersized | 64 B, 16-byte aligned base | **size only** | must trap, mcause 30 |
| 3 misaligned | 16 KiB at base + 8 | **alignment only** | must trap, mcause 30 |

Result (`evidence/result-lines.txt`): **`tohost = 5`, i.e. TESTNUM 5 — both the undersized region
and the misaligned base were sealed**, with the control arm passing. The pre-existing `sealing.S`
was run in the same session and passed at 424 cycles, which is the harness control: it shows the
accept path and the test infrastructure both work. Neither run is near the timeout.

This outcome was **predicted from the netlist before the run** and both arms came back as
predicted, which is what turns a source reading into a measurement.

## Scope, and what is NOT claimed

**Every bitstream built from this tree carries it.** The generated artifact is byte-identical
(`sha256 b594100a051c5030a8c1f7b53c60642ed44b5502728c2bbfa1483a62006e4bfc`) across the working
tree and both synthesis archives, and the source line has not changed since `304fb5080`
(2025-05-01).

**No current caller can trip it.** The monitor's only `SEAL` is `sbi_capstone.c:902`, on a region
of `DOMAIN_DATA_SIZE = 16 * 96` = 1536 bytes (`:187-188`), growing to 2048 for a 2 MiB domain — and
`sbi_capstone.c:730-732` already says in-source that the seal region *"only grows (2048 B), staying
above SEAL's 1024-byte minimum."* Base alignment holds via the granule-aligned `split_size`. So
this is **latent**: real, in silicon, with no exploit path from the monitor as it stands today.

**Verified in simulation only.** Nobody has executed `SEAL` with an undersized or misaligned
region on the board.

**The cause is now LOCALISED to an Anvil precedence rule, and the defect is in the RTL source.**
An earlier version of this report proposed that Anvil binds logical and bitwise operators tighter
than comparison operators generally. **That model is too broad and is refuted in-tree**, by
`capstone_flu_unit.anvil:182` (`SHRINK`): an unparenthesised `==`/`!=` chain next to `||` that
generates **correctly** as three separate comparisons OR'd together
(`capstone_flu_unit.anvil.sv:2328-2338`). Equality binds tighter than `||`, exactly as in C.

The narrower rule fits every observation: **relational operators `<` `>` `<=` `>=` have the
LOWEST precedence in Anvil**, below `||`/`&&`, which in turn sit below `==`/`!=`, which sit below
`&` (the `perm&3'd6!=3'd6` sibling at `:165`, which also generates correctly). So `a < b || c`
parses as `a < (b || c)`; with `b` a nonzero constant the disjunction folds to `1` and the guard
becomes `a == 0` — precisely the collapse above.

This makes it a **source** defect: the line relies on C precedence that Anvil does not use. It has
not been confirmed against an Anvil precedence table, because there is no Anvil compiler in the
tree (`ANVILC ?= anvil`, not present) — reading the upstream table, or running `anvil` on a
five-line case, would close the last gap.

**The class has one other instance, and it is not in SEAL.** Swept in the generated netlist
rather than in source — a source-line grep cannot settle it, because a multi-line unparenthesised
condition evades it by construction. Searching every `core/*.anvil.sv` for the collapse signature
(a `||`/`&&` with a multi-bit operand, or a relational with exactly one 1-bit operand, scoped per
module) returns **7 signature lines in exactly 2 guards**: the line above, and
`capstone_dom_switcher.anvil:115`, whose guard collapses to `cur_idx == 0`. That
one is latent (its enclosing branch is unreachable on this RTL) and is **not part of this issue** —
it is written up separately at
`capstone/docs/history/23-08-2026_22-40-00_anvil-relational-precedence-dom-switcher.md`.
Everything else in the tree already parenthesises its relationals, which is why the class stayed
invisible.

**Either way the fix here is the same one line of parentheses.**

**Which minimum is authoritative is also unresolved**, and matters before anyone "fixes" the RTL:

| where | minimum |
|---|---|
| `capstone-academic-spec` and `capstone-academic-spec` | **1024 B** |
| `capstone-qemu/target/riscv/cap.h:35` | `16 * 33` = **528 B** |
| `capstone-c/samples/capstone.h:36` | **36 B** as written |

Enforcing 1024 in the RTL would start faulting the reference C runtime's own samples.

Separately, `capstone-qemu/target/riscv/op_helper.c:1007-1010` computes this condition and then
**never calls `riscv_raise_exception`** — the block is a debug print — while the three preceding
checks all raise. So the emulator does not enforce it either, by a different mechanism.

## Why no test caught this

`verif/tests/custom/capstone/sealing.S` is the only dedicated `SEAL` test. It seals a
`.zero 4096*4` = 16384-byte region and exercises the accept path only; eleven other tests issue
`SEAL` as setup for CALL/RETURN. **There was no negative test anywhere** — nothing had ever created
the rejecting condition. `evidence/seal-minsize-align.S` is both that missing test and the positive
control `sealing.S` has never had.

It is deliberately **not** registered in `testlist_capstone.yaml`, so an expected failure cannot
redden the regression gate — the same treatment `s06sec-amo-no-resurrect.S` gets for the open AMO
residual, and for the same reason.

## Reproducing

```
cp evidence/seal-minsize-align.S capstone-ariane/verif/tests/custom/capstone/
# add a testlist entry modelled on the `sealing` entry, then:
cd capstone-ariane/verif/sim
rm -f out_*/veri-testharness_sim/seal-minsize-align*
python3 cva6.py --testlist=../tests/testlist_capstone.yaml --test seal-minsize-align \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000
```

Read `tohost` from the `.log.iss`: **1** = control arm trapped (harness broken, nothing else is
interpretable), **2** = only the undersized region accepted, **3** = only the misaligned base
accepted, **5** = both accepted, **4** = a trap with an unexpected `mcause`. A pass means SEAL
enforces the spec.
