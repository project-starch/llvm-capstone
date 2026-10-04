# The memcached allocator-defect corpus on the capability arms, committed at last (2026-10-04)

**Question.** The corpus `bug-corpora/memcached/allocator-repros` has five upstream defect cases and its
`case.json` files record capability results in free-text prose. But **it commits no result bundle of any
kind**: `allocator-repros/.gitignore:4` gitignores `results/` by policy — *"Run summaries stay outside
the repository; the case files' status carries the outcome."* The consequence found in the 2026-10-04
audit is that the paper's memcached row has **no committed backing anywhere**, and
`bug-corpora/INDEX.md` already flags that the corpus *"commits no result bundle of its own"*.

This bundle supplies one, without touching the corpus's policy: it lives in the port the corpus names
(`corpus.json` → `capstone/ports/memcached/allocators`), which does not gitignore `results/`.

## Verdict

**10 of 10 arms as expected, and the oracles are proven able to fail.**

| case | defect | spatial (mode 0) | sublet (mode 1) |
|---:|---|---|---|
| 0 | `rbuf-copied-after-cache-free` | complete | **fault, cause 24**, `pc = 0x1018cb928` |
| 1 | `io-walk-reads-freed-link` | complete | **fault, cause 24**, `pc = 0x1018abca0` |
| 2 | `tail-repair-frees-referenced-item` | complete | **fault, cause 24**, `pc = 0x1018aac68` |
| 3 | `refcount-overflow-frees-linked-item` | complete | **fault, cause 24**, `pc = 0x10186af14` |
| 4 | `unlocked-refcount-drift` | complete | **fault, cause 24**, `pc = 0x10186ac84` |

In every sublet row `pc == expected_pc`, so the fault is at the labelled read probe and not at some
later unrelated access. Every spatial row reports `completed=1`.

**One binary per case, both arms.** Verified by hash, not by reading the runner's docstring: for each
case the `defects.dom` sha256 in the spatial manifest equals the one in the sublet manifest (all five
checked). The arm is a runtime mode argument, so each pair differs in **exactly one thing**.
`SHA256SUMS` carries the five image hashes, the loader, the emulator and the compiler.

## The negative control, which is why the 10/10 means anything

`run-defects.py --negative-control` corrupts the input so the domain refuses the case and then
**requires every arm to be reported failing**; it inverts the exit status, so 0 means every oracle
fired. Its own words: *"A suite whose oracles cannot say FAIL proves nothing by saying PASS."*

    negative control: runner exit 0 -- 10 rows, passed=True: 0, passed=False: 10

So all ten oracles can produce a FAIL, and none of them is vacuous. Without this the 10/10 above would
be an untested pass.

## Two things that went wrong, both recorded rather than smoothed over

1. **The first attempt died before booting anything.** `run-defects.py` defaults `--domain-build` to
   `<tmp>/memcached-allocator-repros/domain`, and I had built to
   `<tmp>/memcached-allocators/build/capstone-domain`, so it raised `FileNotFoundError` on
   `defect-00.dom`. Worth naming because **the harness reported exit code 0 while the runner had exited
   1** — the background notice carries the wrapper's status, not the runner's, so the log is what
   decides.
2. **Case 4's sublet arm needed a second boot.** The first one produced a 3264-byte serial log that
   stops in the OpenSBI banner — it never reached Linux, let alone the guest command. The runner
   classified it correctly as **exit 75, "BOOT PRODUCED NO RESULT"**, which is an infrastructure verdict
   and deliberately *not* a FAIL, because the boot produced no information about the case. Re-run alone,
   it passed as expected (`cause=24`, `pc == expected_pc`). The row in `result-lines.txt` is from that
   second boot.

Also benign, and explained so nobody reads it as breakage: the log contains `pexpect` **EOF
tracebacks** on the sublet arms. A sublet arm halts the domain, QEMU exits, and the prompt wait hits
EOF. Every one of them is followed by the runner's own `OK … sublet … halted` line.

## What this does and does not say

- **It does** give the memcached corpus's capability arms a committed, hash-anchored bundle for the
  first time, with a negative control in the same suite.
- **It does not** re-measure PoisonCap or CheriBSD. Those were last run 2026-09-21 and the cases say
  why: that platform — SDK, purecap sysroot and image — is **not present on this host**. Their record
  remains the `case.json` prose, and this bundle does not change it.
- **It does not** make these defects live at the pin. Four of the five are fix-reversals; only case 2 is
  `live_in_pin: true`, and even there the cited sha changed a *default* rather than the code.
- **It does not** say anything about silicon. Cause 24 is capstone-qemu reloading a revoked capability
  untagged; the deployed silicon lets a stale data access retire (ISSUES Q-11).
- **N = 1 per cell** for the main suite and for the control.

Files: `result-lines.txt` (every row above, main suite and control), `SHA256SUMS`.

Reproduce:

    bug-corpora/memcached/allocator-repros/shared/build-cases.sh capstone-domain <out>
    cmake --preset linux-guest && cmake --build <linux-guest build>     # in ports/memcached/allocators
    runners/capstone-domain/run-defects.py <dir> --domain-build <out> --modes spatial,sublet
    runners/capstone-domain/run-defects.py <dir2> --domain-build <out> --modes spatial,sublet --negative-control
