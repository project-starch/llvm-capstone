---
name: rtl-sim
description: Run the Capstone CVA6 core in Verilator RTL simulation to see what the silicon actually does — every load's address AND returned value, every store's address and data. Use for any "the board says X and we cannot see why" question, to A/B two RTL revisions, or before spending a board boot on something a directed test could answer in 14 seconds. Also read before interpreting a simulation that produced no result.
---

# Running the Capstone core in RTL simulation

The board gives **one number per boot** and every in-frame instrument perturbs what it
measures. Simulation gives a full instruction+memory trace in ~14 s. Prefer it for anything
a directed assembly test can express.

It does NOT replace the board. A bare-metal test runs in M-mode with no monitor; the real
workloads run **inside a capability domain** after `capenter`, on a monitor-carved stack,
reaching globals through a cap table. A clean simulation of a synthetic test is therefore
**not** exoneration — see "Reading a negative result".

## Run one test

```bash
cd capstone/capstone-ariane
docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
  -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir \
  -e NUM_JOBS=16 -e VERILATOR_THREADS=1 cva6-build-rv -c '
set -e; cd /workdir
source verif/regress/install-verilator.sh >/dev/null 2>&1
source verif/regress/install-spike.sh     >/dev/null 2>&1
source verif/sim/setup-env.sh             >/dev/null 2>&1
cd verif/sim
python3 cva6.py --testlist=../tests/testlist_capstone.yaml --test <NAME> \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000'
```

A test is an `.S` in `verif/tests/custom/capstone/` plus an entry in
`verif/tests/testlist_capstone.yaml` (copy the `stc` entry). Add `-e TRACE_FAST=1` for VCD or
`TRACE_COMPACT=1` for FST — either forces a model rebuild (~3 min); unchanged, the model is
reused and a run is ~14 s.

**Never iterate via `run_capstone_tests.sh` or `verif/regress/capstone_tests.sh`.** They end in
`make -C ../.. clean`, and root `Makefile:820-823` deletes `$(ver-library)` — a full
re-verilation every run. Call `cva6.py` directly, as above.

## THREE TRAPS THAT PRODUCE A WRONG VERDICT

These are not inconveniences; each one has been read as a result.

1. **DELETE THE ARTIFACTS BEFORE EVERY RUN.**
   ```bash
   rm -f verif/sim/out_*/veri-testharness_sim/<NAME>*
   ```
   A failed *compile* leaves the previous run's `.log`/`.iss` in place, and the parser happily
   reports them. A stale log has already been read as "exception + timeout" when the run under
   test never built. If you skip the delete, check the file mtime against the wall clock.

2. **`.S` FILES GO THROUGH THE C PREPROCESSOR — including comments.** Writing `CAPTYPE(...)`
   or any other `MACRO(...)` form inside a `#` comment expands as a macro invocation and breaks
   the assembly. Write "CAPTYPE with CAP_TYPE_UNINIT", never `CAPTYPE(CAP_TYPE_UNINIT)`.

3. **`SUCCESS` AT THE TIMEOUT IS NOT A PASS.** The harness prints
   `*** SUCCESS *** (tohost = 0) after N cycles` even when nothing ever wrote `tohost`. If
   `N` equals the `+time_out=` value (2000013 for 2000000), the test **hung or trapped**.
   Compare against a known good cycle count, and grep the `.iss` log for `Exception:`.

## Reading the output

Everything lands in `verif/sim/out_<YYYY-MM-DD>/veri-testharness_sim/`:

| file | what it holds |
|---|---|
| `<name>....log` | **the RVFI trace** — this is the point of the exercise |
| `<name>....log.iss` | raw sim stdout: verdict, `Exception:` lines, all `$display` |
| `<name>....vcd`/`.fst` | waveform, only with `TRACE_FAST`/`TRACE_COMPACT` |

The RVFI trace (`corev_apu/tb/rvfi_tracer.sv:105-120`) records, per retired instruction:

```
3 <pc> (<insn>) x 5 0x0000000000000001 mem 0x0000000080003010   <- a LOAD: rd value, then address
3 <pc> (<insn>) mem 0x0000000080003010 0x0000000000000002       <- a STORE: address, then data
```

So a load's **returned value** and a store's **written data** are both directly readable — which
is exactly what no board instrument can give. Extract with python, not grep (`grep` here is
ugrep and goes silent on control bytes).

`CAPPRINT(reg)` lowers to a `$display` printing the full capability (cursor, revnode id, type,
perm, bounds) into the `.iss` log, and every capability exception self-reports with a cycle
number (`UNEXPECTED_CAP_TYPE`, `OUT_OF_BOUNDS`, …). `CHK_START`/`CHK_END` are marker nops for
bracketing a region.

There is **no raw memory image dump**: the VCD carries `i_sram`'s ports but not its array
(Verilator's `--trace-max-array` default is 32) and nothing calls `$writememh`. Use the RVFI
store/load trace instead; it answers "what did memory hold" without a patch.

## A/B across RTL revisions

Two revisions differing only in `core/**` need no re-clone. **Never `git checkout` in the
submodule** — it destroys uncommitted work. Use a worktree:

```bash
cd capstone/capstone-ariane
git worktree add --detach /path/to/wt <REV>
cp -al tools /path/to/wt/tools                       # hardlink the built toolchain
for d in $(git config -f .gitmodules --get-regexp path | awk '{print $2}'); do
  [ -e "$d/.git" ] && { rm -rf "/path/to/wt/$d"; cp -al "$d" "/path/to/wt/$d"; }
done
cp -al verif/tests/riscv-tests /path/to/wt/verif/tests/riscv-tests   # else the test won't compile
docker run --rm -v /path/to/wt:/workdir --user "$(id -u):$(id -g)" --entrypoint make \
  -e ANVILC=/usr/local/bin/anvil -e CVA6_REPO_DIR=/workdir cva6-build-rv -C core/anvil_build
```

Check `git diff --stat <REV> HEAD` first: if no submodule pins moved, hardlinking is safe.

**Run the IDENTICAL test on both sides.** A comparison of a four-arm run against an eight-arm
run was once reported as a revision difference; it was not one.

## Bring-up, if `cva6-build-rv` or `tools/` is missing

Four things, in order, and each has a wrong answer that wastes an hour:

1. **Anvil is mandatory** — `.anvil.sv` is gitignored (`.gitignore:64`), so the compiler must
   run. It is **not** in `caplifive-build`; it is in `corank/cva6-anvil-build` at
   `/usr/local/bin/anvil`. Generate with `make -C core/anvil_build` (~30 s).
2. **Verilator must be exactly 5.008** — `verif/sim/cva6.py:1033` hard-gates it and refuses
   5.024. Build via `verif/regress/install-verilator.sh`, then **`make -j4`**: at `-j8` or
   higher, `V3Ast.o` gets OOM-killed regardless of free RAM.
3. **Submodules** — 18 of them, plus vendored Spike (the testharness links `-lfesvr -lriscv
   -ldisasm` from `$SPIKE_INSTALL_DIR/lib`, so Spike is needed even for `veri-testharness`).
4. **Toolchain** — stock `gcc-riscv64-unknown-elf` suffices. The Capstone opcodes are raw
   `.insn` directives (`verif/tests/custom/capstone/asm_insn.h`), not named mnemonics, so no
   custom toolchain is required.

## Reading a negative result

A directed test that passes proves only that *that* test does not trigger the defect. Before
concluding anything, confirm from the RVFI trace that the test actually exercised the construct
— count the loop iterations and check the addresses touched. A test that silently did nothing
also passes.

When a board-observed effect does not reproduce, suspect **fidelity** before suspecting the
hypothesis. The known gaps, in rough order of importance: bare M-mode vs a real capability
domain; a register-resident capability vs one loaded from the cap table; a `.data` buffer vs a
monitor-carved stack; iteration count and cache warmth.
