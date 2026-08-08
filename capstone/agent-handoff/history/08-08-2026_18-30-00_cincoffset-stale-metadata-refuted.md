# Stale capability metadata does NOT leak through integer ops — REFUTED, 2026-08-08

**Verdict: the leakage hypothesis is dead.** Ordinary `lui`+`addi` over a register that
previously held a real capability **does** clear its metadata tag. This closes a suspected
R-14 root cause and kills a candidate explanation for the SQLite blocker.

## The result

`verif/tests/custom/capstone/cincoffset-stale-metadata.S`, Verilator, ~13 s:

    *** SUCCESS *** (tohost = 0) after 525 cycles

**The PASS is meaningful because the test carries its own positive control and the control
fired.** It first feeds `CINCOFFSET` a genuinely-tagged capability as rs2 — which
`capstone_flu_unit.anvil:30` must reject — and bails to `selfcheck_fail` if that does not
trap exactly once with `mcause == 25`. The trace shows **exactly one exception and one
`mret`, with mcause 25 (`0x19`)**. So the check can fire, and the trap/mcause plumbing
works. With that established, the real test — rebuild the capability, overwrite the register
with `lui`+`addi`, then use it as `CINCOFFSET`'s rs2 — did **not** trap.

Without that self-check this would have been another clean result from an instrument never
shown able to produce the opposite, which is the most expensive mistake on this project. The
test's author built the control in; running it is what made the negative worth anything.

## What it refutes

The chain reasoned out in the test's own header — `commit_stage.sv:279/325` gates the
metadata regfile's write-enable while the integer regfile writes unconditionally, and
`ariane_regfile_ff.sv` merely *holds* the old word when write-enable is low, so the tag
should survive — **does not produce an observable stale tag on this RTL.** Whatever the
mechanism, the tag is gone by the time `CINCOFFSET` reads rs2.

## What it cost, and why the reasoning was seductive

A real structural difference had been found behind the SQLite blocker:

| image | `__capstone_cap_init` | `cincoffset` | `cincoffsetimm` |
|---|---|---|---|
| `sqlite_silicon.dom` (wedges) | 1522 insns | **254** | 96 |
| `fdp0.dom` / fdreg model (runs clean) | 47 insns | **0** | 0 |

Combined with `capstone_flu_unit.anvil:29-34` raising `UNEXPECTED_OPERAND` when rs2 carries
capability metadata, and R-5's "illegal capability ops wedge rather than trap", this was a
coherent story tying the blocker to the same ungated-metadata path as R-18/R-19
(`issue_read_operands.sv:1140`).

**The structural difference is real. The proposed reason it would matter is not.** The
mechanism required an integer offset register carrying a stale tag, and that register is
clean.

## Scope — narrow, and stated so it is not over-read

The run exercises the **`lui`+`addi` producer only**, which is the R-14 codegen signature. A
register whose integer value arrives by a **load**, an **`add`**, or a **register copy** is
NOT covered. If the stale-tag idea is ever revived it must be revived for one of those
producers, with the same self-check discipline.

## Reusable invocation

Reconstructing this was itself a task; it is now recorded in the test header. ~13 s against
the prebuilt model at `work-ver/Variane_testharness` — no rebuild.

    cd capstone/capstone-ariane
    docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
      -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir cva6-build-rv -c '
        cd /workdir; source verif/sim/setup-env.sh >/dev/null 2>&1; cd verif/sim
        python3 cva6.py --testlist=../tests/testlist_capstone.yaml \
          --test <TESTNAME> -o out_X --iss_yaml cva6.yaml \
          --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
          --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE'

To pass a define to a test, add `--gcc_opts="-DFOO=1 "` — **the trailing space is required**,
the concatenation at `cva6.py:1252-1253` has no separator.

## Still open on SQLite

The wedge remains localised by the surviving clamp series (`n1` returns after
`zName = aDef[i].zName`; `n2` does not return after `sqlite3Strlen30(zName)`). The two
walkers compile to near-identical code — same prologue, same `*144` indexing, same
`ldc a1, 0x70(a0)`, same `auipc/addi/jalr` call sequence, same `auipc` count. Two further
hypotheses died the same day: "SQLite uses `auipc` where fdreg does not" (both use two) and
"the call overruns the PCC/code window" (the target is inside the code region).

The cap-init descriptors were also checked and are **well-formed** — 179 records, no
non-power-of-2 alignments, no zero sizes, no init source outside the file, one 256 KiB record
which is the SQLite heap. A malformed descriptor is not the answer either.

The submodule commit carrying the test result is `57243ede4`; it **cannot be pushed** (the
RTL remote returns 403), which is why this note exists here.
