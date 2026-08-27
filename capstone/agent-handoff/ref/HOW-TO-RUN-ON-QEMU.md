# How to run a program under Capstone on QEMU, and how to prove a boundary bug is caught

Written for a collaborator bringing a **new workload** (e.g. the cross-language FFI
corpus) into the Capstone runtime. It is deliberately self-contained and uses only
what is already in the tree — it does **not** depend on the compiler/ABI/board work
currently in flux, so churn there cannot block you.

Everything below runs under **QEMU**. You do not need the FPGA board for any of it.

---

## 0. Setup, once

```bash
cd <repo root>
source capstone/tests/capstone-test-env.sh
```

That exports `CAPSTONE_CLANG`, `CAPSTONE_LD_LLD`, `CAPSTONE_LLVM_BIN`,
`CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_TMP_ROOT` (`/tmp/capstone`). Every script below
assumes it. **Each new shell needs it again** — a surprising number of confusing
failures are just an unset `$CAPSTONE_CLANG` silently producing no output.

---

## 1. The execution model — read this before writing code

A Capstone run has **two halves**:

- a **host controller**, an ordinary Linux/RISC-V program (`*.user`) built with the
  buildroot cross-gcc. It creates the domain and reads results back.
- a **domain**, a freestanding program (`*.dom`) built with the Capstone clang. It runs
  under capability confinement and cannot make syscalls directly.

They communicate through a **shared region**. The single most important fact, and the
one that has cost the most time here:

> **The annotated share IS the domain entry.**

`shared_region_annotated(dom_id, region_id, ...)` *enters the domain* with that region
as its argument. Do **not** call `call_dom()` afterwards — that enters a second time
through a different path whose first argument is only an 8-byte return slot, and your
domain will fault reading anything past it. Four separate attempts failed this way
before it was understood.

Corollaries:
- **N shares = N entries.** Sharing two regions enters the domain twice. That is how a
  workload gets more than one channel (see `capstone-reentry.c`).
- Map and zero the region **before** sharing it.

Your domain's entry point is `void domain_main(unsigned *res, unsigned func)`; write
results into `res[]`, and the host reads them out of the mapped region.

---

## 2. Build and run a domain

The generic path:

```bash
bash capstone/tests/runtime-qemu/build-domain.sh <domain_main.c> <out.dom>
python3 capstone/tests/runtime-qemu/run-domain-smoke.py <out.dom>
```

Useful `run-domain-smoke.py` flags:

| flag | what it is for |
|---|---|
| `--guest-command` | the shell command to run inside the guest |
| `--success-marker` | a regex the run must print; this is your PASS criterion |
| `--domain-loader` | use a different host controller than the default |
| `--share-dir` | host directory exposed to the guest over 9p as `/mnt/host` |
| `--log-file` | where the full serial log lands |
| `--timeout-multiplier` | raise for long workloads |

A complete worked example, with both halves and a real pass criterion, is
`capstone/benchmarks/sqlite/run-sqlite-silicon.sh`. **Copy its shape.** It builds both
halves, runs them, and requires five specific markers.

> **Changed 2026-08-20, and the old pointer would have cost you a build.** This used to name
> `run-sqlite-memory.sh`, which **no longer runs**: its domain is 3.34 MB and the module has
> doubled the allocation since `caplifive-buildroot` `37ed834` (2026-08-12), which halved the
> largest creatable domain to 2.00 MB of code. It dies at `create_dom` before any SQL, with
> `Failed to allocate memory for domain.` in the guest `dmesg` (see `ISSUES.md` Q-01).
> `run-sqlite-silicon.sh` is the better model anyway — it builds in the **silicon configuration**,
> so what you test under QEMU is what the board runs.
>
> **Two traps it documents that any copy must keep.** Resolve and export `OUT_DIR` **before**
> invoking the build scripts — they each default to a *different* directory, so a late `OUT_DIR`
> silently splits the domain and the host across two trees. And read both the `.dom` **and** the
> host from that same `OUT_DIR`: the host links `libcapstone`, which packs the globals offset into
> `entry_offset`, so a mismatched host runs the wrong geometry — either the loud `0xB10B`
> blob-does-not-fit error or, worse, a plausible run of the wrong binary.

---

## 3. Proving a boundary bug is CAUGHT — the part that matters

For a memory-safety claim it is not enough that the program runs. You have to show the
defense fires on the bug and does **not** fire without it. The pattern used throughout
this repo is a **matched pair**:

| variant | what it does | expected outcome |
|---|---|---|
| **fault** | the real bug: object handed across the boundary, revoked, then used | domain **faults**; QEMU exits; the harness returns **non-zero BY DESIGN** |
| **control** | identical program, revoke removed (e.g. `-DFOO_NO_REVOKE`) | domain **returns normally** with a correct value |

Both are required. The control is not optional politeness — it is what distinguishes a
real revoke from an unrelated fault. Concretely: at `-O0` a plain spill/reload can also
produce a "tag gone" cause-24 fault, which looks identical to a caught use-after-free
until the control shows the same program completing when the revoke is removed.

Read `capstone/benchmarks/sqlite/run-sqlite-row3.sh` — its header comment states this
contract exactly, and `sqlite_row3_domain.c` shows the wrapper that carves an
independently revocable copy and revokes it at the right moment.

**The evidence you keep** is the monitor's fault line from the serial log, e.g.

```
[CAPSTONE] domain halted by capability fault: cause = <N>, pc = 0x..., badaddr = 0x...
```

plus the control's clean return. Quote both in your write-up.

### How to read that line, because two of its three numbers lie

**`pc` is the entry PC of the translation block, NOT the faulting instruction.**
Capability faults are raised from `_helper_access_with_cap` in
`capstone-qemu/target/riscv/op_helper.c`, which is `static` and is not inlined
(`nm build/qemu-system-riscv64 | grep with_cap` shows it as a local `t` symbol). So
`GETPC()` there returns an address inside `helper_load_with_cap` /
`helper_store_with_cap`, which is not in the code-gen buffer, and `cpu_restore_state`
takes its documented early exit without ever restoring `env->pc`. What gets printed
is whatever TCG last wrote to `cpu_pc`.

In practice the reported pc has landed on the return address of a `jalr` three times
running, with the real faulting instruction +4 and +12 further on. **There is no fixed
offset.** Read forward from the reported pc to the first access whose width matches the
cause, and treat the pc as a starting line, not an answer. This project has retracted
three conclusions drawn from that number.

**`badaddr` / `tval` are stale for capability faults.** The raise sites in
`op_helper.c` never assign `env->badaddr`, while `cpu_helper.c` reports
`tval = env->badaddr`. In one run the emulator printed `addr = 0x1022d1e28` while the
fault line said `badaddr = 0x1018ad346`, which is not even 4-aligned and therefore
cannot be the address of a 16-byte access.

**The `[CAPSTONE] Unaligned cap access (addr = ...)` line is the one that tells the
truth**, and it is also what distinguishes a capability misalignment from an ordinary
one: cause 4 is `LOAD_ADDR_MIS`, and the capability path raises it only inside
`if (size == 16)`, for loads and stores alike, while `riscv_cpu_do_unaligned_access`
raises the same code for plain loads and *does* set `badaddr`.

The cause code is the reliable part. It narrows the instruction by WIDTH, which is
often enough on its own: a two-byte load cannot raise a cause that only a 16-byte
access can.

A smaller, purely synthetic example to start from:
`capstone/tests/runtime-qemu/build-borrow-revoke-uaf-probe.sh` and its `run-` sibling.

---

## 4. Traps that have cost real time here

- **Exit code 75 means INFRA FLAKE, not failure.** `run-domain-smoke.py` returns 75 when
  QEMU dies before the guest login prompt. Retry up to 3× before believing a failure.
- **Gate on exit status, never on grepping output for error strings.** This has produced
  wrong conclusions at least three times, in both directions — a "failure" that was the
  Makefile echoing its own recipe, and a "success" that was a grep finding nothing
  because the file was binary.
- **Serialize QEMU runs.** All suites share one `rootfs.ext2` write lock. Never run two
  at once; you will get spurious EXT4 errors that look like real corruption.
- **A faulted domain poisons later `create_dom` in the same guest session.** Run the
  fault variant and the control as **separate** QEMU boots, one domain each.
- **Verify by dumping, not by reading.** If you believe a table/section/value is what you
  think it is, print it. Roughly ten plausible hypotheses have been refuted here by
  dumping the artifact rather than reasoning about it.
- **`-O0` vs `-O1` changes behaviour**, sometimes decisively. State the level in every
  result; do not compare numbers across levels.

---

## 5. What to hand back

For each case: the two build/run commands (fault + control), the fault line from the
serial log, the control's return value, and the `-O` level. If a case does **not** fault
when you expected it to, that is a result worth reporting, not a failure to hide — the
uncaught cases are as informative as the caught ones for the paper.

Put shared artifacts under `capstone/benchmarks/<your-corpus>/` following the `sqlite/`
layout: `build-*.sh`, `run-*.sh`, `*_domain.c`, and a `README.md` stating what is
verbatim from the original bug and what was adapted.
