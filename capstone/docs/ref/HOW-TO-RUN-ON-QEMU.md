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
`capstone/ports/sqlite/run-sqlite-silicon.sh`. **Copy its shape.** It builds both
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

Read `capstone/ports/sqlite/run-sqlite-row3.sh` — its header comment states this
contract exactly, and `sqlite_row3_domain.c` shows the wrapper that carves an
independently revocable copy and revokes it at the right moment.

**The evidence you keep** is the monitor's fault line from the serial log, e.g.

```
[CAPSTONE] domain halted by capability fault: cause = <N>, pc = 0x..., badaddr = 0x...
```

plus the control's clean return. Quote both in your write-up.

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

- **The loader stalls intermittently, and it looks exactly like a hang in your domain.**
  Signature: the serial log stops after `Segment size = ...` and never prints
  `Loadable size`, while QEMU spins at 100% CPU. That gap is `mmap` + `memset` +
  `memcpy` from the 9p-mapped image inside `capstone-test.user`, i.e. before the domain
  is created and before any of your code runs. Roughly half the boots in one session
  did this, on images of about 1.6 MB. Cause unknown; do not read it as a result.
  Retry, and check WHERE the log stops before believing anything about the domain.
  (An earlier note here blamed a second load in the same boot. That is refuted: the
  stall happens on the first load too.)

---

## 4b. Locating a fault: `tests/runtime-qemu/fault-locate.py`

A capability fault used to take twenty to forty minutes to place: read the anchor
rung's return value, subtract the symbol address for the load base, subtract that from
the pc, `llvm-nm`, read the disassembly, guess which struct field an offset names. It
produced wrong readings more than once. The script does all of it:

```bash
python3 capstone/tests/runtime-qemu/fault-locate.py <run-log> <image.dom>
```

It prints the load base, the image offset, the enclosing function, the **faulting
instruction** and the source line with its inline chain.

Two things it does that are not arithmetic:

- **It checks the pc, then disambiguates it.** The monitor restores the exact pc since
  `_helper_access_with_cap` was given the caller's return address (before that its
  `GETPC()` named QEMU's own text, `cpu_restore_state()` failed silently, and `env->pc`
  kept the translation-block ENTRY). The script takes the instruction at the reported pc
  when it matches `rs1`/imm/size and says `the reported pc is exact`; otherwise it scans
  forward and says how stale the pc was. Reading an old log still works.
- **It REFUSES rather than guessing.** No fault line, no anchor, a load base whose
  offset leaves the executable section, no matching instruction in the window: exit 2
  with what it looked for. A tool that prints an empty result reads like a finding.
- **It refuses AMBIGUITY too, and that is not theoretical.** 33.9% of the memory
  instructions in a real image have another instruction with the same
  (base, immediate, size) inside the 256-byte window, 39.7% for faults that report no
  size, and `-capstone-double-ldc` emits such pairs by construction. Naming the first
  candidate put the answer in a *different function* 11.3% of the time. So when the pc
  is not exact and more than one candidate matches, it lists them and exits 2.

Source lines need line tables. For the mruby port that is `MRUBY_DEBUG_INFO=1`, which
adds `-gline-tables-only`. Debug sections are non-alloc, so the twin is usable as an
oracle for the image that actually ran — but **check** that rather than assume it, and
check more than the bytes. Identical `.text` CONTENT with a different `.text` ADDRESS
would still give wrong line numbers:

```bash
# 1. same code
llvm-objcopy -O binary --only-section=.text <image> - | md5sum        # both images
# 2. same addresses
llvm-readelf -SW <image> | grep -E '\.text|\.rodata|\.data '          # both images
# 3. same symbols
diff <(llvm-nm -n <release> | awk '$2=="T"{print $1,$3}') \
     <(llvm-nm -n <twin>    | awk '$2=="T"{print $1,$3}')
```

All three passed for the mruby pair the tool was first used on; the md5 alone is the
weaker test.

`--self-test <image.dom>` runs four controls built out of the image itself: an exact pc
must be taken as-is, a stale pc must still resolve, an ambiguous window must be REFUSED,
and an immediate proven absent from the scanned window must be refused. The negative
controls are built against the window the tool will actually scan and assert their own
premise, because the first version of them was tautological: it built the impossible
immediate out of the same window it was about to search, so the refusal was guaranteed
by construction and the control proved nothing.

---

## 5. What to hand back

For each case: the two build/run commands (fault + control), the fault line from the
serial log, the control's return value, and the `-O` level. If a case does **not** fault
when you expected it to, that is a result worth reporting, not a failure to hide — the
uncaught cases are as informative as the caught ones for the paper.

Put shared artifacts under `capstone/benchmarks/<your-corpus>/` following the `sqlite/`
layout: `build-*.sh`, `run-*.sh`, `*_domain.c`, and a `README.md` stating what is
verbatim from the original bug and what was adapted.

## Rebuilding the QEMU monitor — one tree, `TARGET=qemu` (since 2026-09-07)

The board and QEMU flavours of `caplifive-buildroot`, the OpenSBI wrapper and the monitor are one
source on `capstone-bootstrap` (`docs/plans/monitor-unification.md`). Both checkouts
(`capstone/caplifive-buildroot` for QEMU, `capstone/caplifive-system/sw/buildroot` for the board)
track that branch; the target is a build variable and each target has its own output directory
(`build-qemu/`, `build-fpga/`) with `build` a per-checkout symlink to the one the checkout serves —
the harnesses keep reading `build/images`.

```bash
cd capstone/caplifive-buildroot            # build -> build-qemu here
export CAPSTONE_CC_PATH="$(realpath ../capstone-c)"   # REQUIRED; every build prints the compiler it used
# 0. still parse the wrapper in place first when the monitor changed (a failed regen deletes the .c.S):
( cd ../capstone-c && cargo run -q -- --abi capstone \
    "$PWD/../caplifive-buildroot/components/opensbi/lib/sbi/sbi_capstone_dom.c" \
    -- -I"$PWD/../caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi" -D__riscv_xlen=64 \
    -DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE > /tmp/x.S ) && echo parses
make TARGET=qemu build A=opensbi-rebuild   # regenerates the .c.S (also when TARGET changed: a stamp of the defines is a prerequisite), relinks fw_jump
make TARGET=qemu build A=capstone-sbi-domain-rebuild   # sbi.dom, the third monitor copy (package/capstone-sbi-domain, still its own checkout)
make TARGET=qemu build                     # repack rootfs
```

`sbi_capstone.c` is ONE file for both targets: per-target code sits under `CAPSTONE_TARGET_FPGA` /
`CAPSTONE_TARGET_QEMU` (`capstone_target.h`; the OpenSBI platform directory supplies the define for
OpenSBI's own compile, `CAPSTONE_EXTRA_DEFS` for the capstone-c regeneration). The gates below still
apply; the "two copies" paragraph now describes only `sbi.dom`'s package copy.

## Rebuilding the stand-in monitor (QEMU flavour) — the pre-unification recipe, kept for its gate lessons

The QEMU firmware's monitor is **not** rebuilt by a bare `make build`, and four different things
make a rebuild silently do nothing. Learned over eight attempts on 2026-09-05 (Q-03); every
failure below was caught by a gate before a wrong binary reached the image.

**Two copies of `sbi_capstone.c`, and the ecall decides which is live.**
`package/capstone-sbi-domain/capstone-sbi/` compiles into `sbi.dom`; `components/opensbi/lib/sbi/
capstone-sbi/` is regenerated into `sbi_capstone_dom.c.S` and linked into `fw_jump.elf`. The
kernel module reaches `DOM_CREATE` via `sbi_ecall` (`modcapstone/module/capstone.c:122`), so
**`fw_jump` — the components copy — is what QEMU executes** for domain creation. Patch both; the
components one is the one that matters. They are two checkouts of `caplifive-sbi` and may sit on
different commits.

```bash
cd capstone/caplifive-buildroot
export CAPSTONE_CC_PATH="$(realpath ../capstone-c)"   # or the .c.S regen runs `cd ""` and fails
# 0. verify the edited source parses BEFORE spending a build -- a FAILED regen rm -f's the .c.S,
#    leaving the tree unable to rebuild fw_jump until the source parses again:
( cd ../capstone-c && cargo run -q -- --abi capstone \
    "$PWD/../caplifive-buildroot/components/opensbi/lib/sbi/sbi_capstone_dom.c" \
    -- -I"$PWD/../caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi" -D__riscv_xlen=64 \
    > /tmp/x.S ) && echo parses      # run on the wrapper IN PLACE: its #include is a quoted path
make build A=capstone-sbi-domain-rebuild   # sbi.dom  (SITE_METHOD=local: rsync + rebuild)
make build A=opensbi-custom-rebuild        # regen .c.S, RE-SYNC it into build/build/opensbi-custom, relink fw_jump
make build                                 # repack rootfs.cpio / rootfs.ext2
```

**Gate every step against the ORIGINAL state, never the previous attempt** — identical source
rebuilds to identical bytes and rsync preserves an unchanged file's mtime, so "did it change since
last run" fires false after the fix has landed:

- `sha256sum build/build/opensbi-custom/build/platform/generic/firmware/fw_jump.elf` differs from the pre-fix sha;
- `sha256sum build/build/capstone-sbi-domain-1.0/sbi.dom` differs from the pre-fix sha, and `llvm-nm` shows the new symbol;
- the regenerated `.c.S` carries the NEW constant and not the old one — **grep the DECIMAL form**:
  Capstone-C emits immediates as `li a1, 4662`, never `0x1236`, so a `grep -c 0x1234` gate reads 0
  on every build and can never fire (it did exactly that on 2026-09-05 and passed a vacuous check
  for two builds). Positive-control the gate: compile the pre-fix source to `/tmp` and confirm the
  same grep fails it;
- `build/images/rootfs.ext2` mtime advanced;
- then `capstone/tests/runtime-qemu/run-smoke.sh` rc 0.

**Capstone-C does not short-circuit `||` or `&&`.** `LogicalOr`/`LogicalAnd` lower to plain integer
`Or`/`And` DAG nodes (`capstone-c/src/dag_builder.rs:1094-1095`, `dag.rs:418`), so EVERY operand is
evaluated. A guard of the form `if(id >= n || table[id] == 0)` reads `table[id]` for an out-of-range
`id` and, since every global is its own exactly-sized capability, takes an M-mode bounds fault. Write
the range check and the table read as two statements. Found 2026-09-05 by the Q-03 consistency
loader's out-of-range share, after the same guard had passed every replay.

**Capstone-C is two pipelines** — components runs real cpp (`--` args), package runs bare
`cargo run` with its own preprocessing — and they have produced different parse verdicts on the
same intermediate source. If a block will not parse, do not theorise about the parser and do not
trust a rule from a previous session: bisect in `/tmp` with the **unpatched source as the
control** (it compiles) and a deliberate syntax error as the negative control. (A rule recorded
here on 2026-09-05 — "an array assignment in the nested tail branch fails under both" — was
retracted the same day when an auditor compiled that exact form cleanly under both.)
