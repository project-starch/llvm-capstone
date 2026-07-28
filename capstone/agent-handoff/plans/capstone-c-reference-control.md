# The `capstone-c` reference control — scoped, not yet run

**Why this is the single most valuable board experiment outstanding.** The project has
asked three times whether R-1 (and R-6/R-8) are hardware faults, and the honest answer is
*we do not know*, for one concrete reason: **our gp-captable ABI is non-standard and has
never been checked against the board owner's reference implementation on the same board.**

Everything we cite as evidence for "hardware" has the same shape — *our* compiler, *our*
ABI, *our* patched QEMU says it is fine, the board disagrees. That is consistent with a
hardware fault **and** with our ABI provoking something a conforming one would not. Our
QEMU cannot arbitrate: it was deliberately patched to be *more permissive* in at least
three places (`delin` idempotence "rather than faulting", bit-preserving untagged
`ldc`/`stc`, gp fabrication).

**The experiment:** build a kernel with the R-1 shape using `capstone-c` — the reference
the board owner pointed at, which uses the cscratch cap-table model — and run it on the
same board.

| outcome | conclusion |
|---|---|
| reference **also fails** | R-1 is the platform's. Confidence goes from "high, not certain" to settled, and the bug report is strengthened enormously. |
| reference **passes** | R-1 is **ours** — an ABI/codegen interaction. That would redirect the whole investigation and potentially unblock 6+ benchmarks. |

Either way it is decisive, which nothing else on the list is.

## What is actually involved (scoped 2026-07-28, not started)

`capstone/capstone-c/` is a **Rust** compiler (`Cargo.toml`, `local_build.sh` which
bootstraps rustup if needed) with C samples in `samples/` — `array.c`, `cap_ops.c`,
`call.c` and others.

Steps, with the unknowns named rather than glossed:
1. **Build `capstone-c`** — `local_build.sh`. Needs a Rust toolchain; unknown whether the
   pinned version is present on this machine.
2. **Write the probe** — port `rawhazard5`'s failing shape (register-indexed load with an
   intervening store through a second capability into the same object) into a `samples/`-style
   C file. Keep it minimal; the shape is five lines.
3. **UNKNOWN, and the main risk: output format.** Does `capstone-c` emit something our
   loader can run (`create_dom` + an entry that writes a result), or does it assume its own
   runtime? If the latter, this needs a small shim, and that is where the time goes.
   **Resolve this before allocating board time.**
4. **Establish an oracle** — the same value computed natively, as every ladder rung does.
5. **Run once on the board.** No probe-delivery risk if the result comes back as the domain
   return value rather than through debug slots (see I-4: newly written probes have failed
   to deliver twice).

## Do this before spending a boot

Step 3 is the gate. If `capstone-c`'s output is not loadable by `capstone-test.user`, the
experiment is a shim-writing task first and a board task second. Find that out off-board.

## Related

- **R-1** — the fault under test.
- **I-4** — do not deliver this probe through debug slots; use the return value.
- **R-2** — already conceded as probably ours, and a precedent for the reference disagreeing
  with our assumptions.
