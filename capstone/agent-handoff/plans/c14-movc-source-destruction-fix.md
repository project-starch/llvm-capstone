# C-14 fix proposal — `movc` destroys its source register

**Status:** proposal, awaiting a decision. Nothing implemented.
**Root cause:** see `ref/ISSUES.md` C-14. Proven numerically 2026-07-30.

## The constraint

RTL `movc rd, rs` nulls `rs` unless `rs` is a `NONLIN` capability
(`capstone_flu_unit.anvil:13-21`). QEMU only nulls it when `rs` is a *tagged*
non-copyable capability (`op_helper.c:580-584`), so scalars survive under QEMU and die on
silicon. `copyPhysReg` emits `MOVC` for every GPR-to-GPR copy
(`CapstoneInstrInfo.cpp:520-523`), so this reaches ordinary compiled code everywhere.

No single instruction is a safe universal copy:

| candidate | scalar source | capability source |
|---|---|---|
| `addi rd, rs, 0` | correct, preserves source | drops metadata -- not a capability copy |
| `movc rd, rs` | **destroys source** | correct for NONLIN; destroys others |
| `cincoffset rd, rs, x0` | RTL preserves rs1, QEMU nulls it (C-4b) | same divergence |
| `cincoffsetimm rd, rs, 0` | traps UNEXPECTED_OPERAND on NOT_CAP | works, but see above |

`copyPhysReg` runs after register allocation and sees only `GPR`, which holds both
scalars and capabilities. It therefore cannot choose correctly today. **That is the whole
problem** -- everything below is a way of getting the type to the copy.

## Options

### A. Separate register classes for capabilities and scalars
The principled fix. Give capabilities their own class so `copyPhysReg` dispatches on
`RegClass.contains(...)` exactly as it already does for `GPRF16`/`GPRF32`
(`CapstoneInstrInfo.cpp:526-536`). Scalar copies become `ADDI`, capability copies stay
`MOVC`.

- Correct by construction, and fixes every future copy site, not just loops.
- Largest change: affects ISel patterns, calling convention, spill/reload, and the
  gp-captable lowering. Not a same-week change, and it touches the ABI the board
  firmware already agrees with.

### B. A copy pseudo selected by type at ISel, expanded late
Emit `PseudoCopyScalar` / `PseudoCopyCap` where the DAG still knows i64 vs i128, and
expand them after RA. `copyPhysReg` keeps MOVC only for the cases it genuinely cannot
attribute.

- Much smaller than A and reversible.
- Does not cover copies the register allocator itself invents (spills, coalescing
  artefacts, two-address fixups) -- which is exactly where the failing `movc a4, a6`
  came from. **This is the main risk and needs checking before committing to it.**

### C. Targeted late peephole: rewrite `MOVC rd, rs` where `rs` is live-after
A MachineFunction pass over the post-RA output: for each `MOVC rd, rs` whose `rs` is live
after the instruction, substitute a safe form.

- Smallest, most auditable, and directly aimed at the measured failures.
- Still needs the scalar/capability distinction to pick the substitute. If `rs` is live
  after the copy, it cannot be a LINEAR capability (duplicating one is illegal anyway), so
  it is either a scalar or NONLIN -- and MOVC is already safe for NONLIN. So the pass only
  has to be right about "is this a scalar", which liveness plus the defining instruction
  can usually answer.
- Verification is cheap: `tests/runtime-qemu/silicon-ladder/check-movc-reuse.py` already
  classifies all 13 measured rungs correctly and reports 444 sites in SQLite.

### D. Ask the board owner to fix the RTL
`movc`'s guard should test "is this actually a capability" rather than "is it NONLIN",
matching QEMU. Worth raising regardless of what we do in the compiler, because the
divergence will bite any future toolchain.

- Right long-term answer, but needs a bitstream rebuild and is not under our control, so
  it cannot be the plan for the deadline.
- Also worth reporting: `check_fwd_rs1` (`ariane_pkg.sv:925-931`) lists
  `{SPLIT, MOVC, CJALR, CCSRRW, STC}` and is **dead code**, while the rs1 write-back is
  actually gated on the much broader `check_cap_op` (`commit_stage.sv:278-281`). That
  looks like an unintended widening and is a second thing for the board owner to confirm.

## Recommendation

**C now, D in parallel, A or B after the deadline.** C is the only one that can be
written, QEMU-gated and board-validated in the time available, and it has a ready-made
verification gate. D costs one short message and may fix it properly for everyone.

## Validation plan for whichever is chosen

1. `check-movc-reuse.py` reports 0 on gpw16, gpn2, gpn4, gpw2 (currently 1 each).
2. Rebuild and re-run those four plus the `beebs_primer1` control in ONE board session.
   Expect all to return their exact oracles; `gpw2`'s true oracle is 3983810698, not the
   `%u` placeholder its old file contained.
3. Only then rebuild SQLite and re-run it.
4. Do not touch the paper's SQLite claim until 3 passes -- it currently says SQLite has
   not run on the board, which remains accurate.
