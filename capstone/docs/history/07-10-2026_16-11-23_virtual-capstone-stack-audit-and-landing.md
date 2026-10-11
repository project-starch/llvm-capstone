# Audit and landing of the virtual-Capstone stack: 16 PRs across two repos

Landed 2026-10-07. QEMU #13-#16 onto `capstone-qemu` `virtual-capstone`; llvm-capstone #182-#193
onto `dev`. Integration targets and ordering were given by the project lead, and the PR base graph
read from the API matched them exactly.

**The one thing to read if you read nothing else:** the new virtual-runtime code paths are **not
qualified** by anything that ran here. See "What was NOT gated" below. The legacy suites prove the
stack did not *regress* existing behaviour; they do not qualify the new path.

## What landed

| repo / branch | PRs | result |
|---|---|---|
| `capstone-qemu` / `virtual-capstone` | #13 #14 #15 #16 | merged, head `b649187a98` |
| `llvm-capstone` / `dev` | #182 #183 #184 #185 #191 #186-#190 #192 #193 | merged, head `be599387fe0a` |

`c128-qemu-merge` is deliberately untouched: it remains physical Capstone's branch.

Merged with `git merge --no-ff`, **not** squashed. Every one of the 16 PRs was authored by the
external collaborator, so squashing would have re-authored their work as the lead's; the rule's
exception for a collaborator's PR exists precisely to keep that authorship. Each PR was already a
single commit, so "one logically complete change per landing" holds anyway. `dev` now carries 11
content commits under the collaborator's name.

A second reason the mechanism mattered: #184 had to be repinned (below). Under `--no-ff` the
merge-base for each later PR *is* the previous merge, so the pin resolved cleanly every time.
Under squash, every PR above #184 would have kept a pre-#184 merge-base and conflicted on
`capstone/capstone-qemu` — seven manual pointer resolutions.

## The submodule pin, which is the trap in this stack

#184 pinned `capstone/capstone-qemu` at `9bf9c1f286` — the head of `virtual-qemu-recycling`, a
*feature branch*. A parent pointing there would reference history not on the integration branch.
It was repinned to the landed merge instead, and then moved again to `b649187a989c` after the fix
below. The RTL, Buildroot and system pins were left exactly as they were on `dev`; exactly one
pointer moves in the whole stack. Verified before pushing that the commit `dev` names is present
on the QEMU remote, and the submodule was pushed before the parent.

## The regression this landing found, and fixed

`linear-uninit-corpus` failed six arms — two probes at -O0, -O1 and -O2:

    FAIL uninit_use_before_init_fault (faulted with cause 26, expected 26 -- wrong reason)

The oracle is `cause == 26 AND the reason line present`. The cause was correct; the reason line was
gone. QEMU #13 rewrote the load-type condition in `_helper_access_with_cap` from
`!is_store && type == UNINIT` to a broader test excluding LIN/NONLIN/SEALEDRET, and routed the
raise through the new `capstone_raise_cap_fault`, which records `capstone_tval` instead of
printing. The site's `CAPSTONE_DEBUG_PRINT` went with it.

**It was collateral, not a design decision:** `CAPSTONE_DEBUG_PRINT` went 49 -> 48 across
`op_helper.c`, at the one site being rewritten. A deliberate move away from printed reasons would
have taken more than one. The comment directly above the deleted line even names the probe that
broke, asserting it "passes only with this check in place" — the check *is* in place; the probe was
failing on the message half. That suite lives in the superproject, outside the QEMU repo's own
validation list, which is how a cross-repo gap like this survives review.

Fixed by restoring the diagnostic (`b649187a98`), with the original wording emitted only for the
UNINIT case it describes and a general message naming the actual type otherwise, since the
condition is now broader. The broadened check itself is correct and untouched.

Evidence, each step checked rather than assumed: the suite was 0 FAIL on 03-10, 24-09 (twice) and
16-09; the expected string is present in the pre-merge binary and absent from the merged one;
attribution is per-commit, not from a range log; and the blast radius was *predicted* before being
measured — only the UNINIT string was lost, `UNTAGGED` and `DROPPED` survived, so exactly the
uninit arms should fail, and exactly those six did. After the fix: 0 FAIL / 21 PASS with the
suite's own marker emitted.

## What was gated, and what the numbers were

| gate | result |
|---|---|
| QEMU build (merged chain) | clean, 1550/1550, binary runs |
| `virtual-capstone-runtime` | 33/33 |
| `virtual-capstone-m1` | 60/60 |
| `trusted-linux-u-access` | 69/69 |
| host/guest node model | 8,949 prefixes each, 6 directed, **3 mutation controls detected** |
| lit arm 1 (CodeGen/Capstone + 17 clang capstone) | **136/136** |
| lit arm 2 (CodeGen/RISCV + Generic) | 2444/2458; the 4 failures are pre-existing |
| core QEMU suites | **15 of 16 PASS** |

13 reject-arms fired across the QEMU suites and 3 mutation controls were detected by the model, so
these are not vacuous greens. The merged QEMU accepts `x-capstone-u-mode` where the pre-merge
binary rejects it — a behavioural check that the chain actually landed, not a path comparison.

Three failure classes, all attributed by a **matched control** rather than by argument:

- **`beebs`** — 5 arms, all `bits/libc-header-start.h not found`, which exists only at the
  multiarch path on this host. Today's failing set is a *strict subset* of 24-09's; zero new. One
  test (`stringsearch1`) has since gone from fail to pass.
- **`lit` / `lit-generic` in the nightly** — die in 0 s with `AttributeError: enable_profcheck`
  whenever `CAPSTONE_REPO_ROOT` is a git worktree: they read the worktree's `llvm/test/lit.cfg.py`
  while the generated `lit.site.cfg.py` lives only in the main clone's build dir. A worktree
  containing *no stack at all* fails identically. Coverage was recovered by running both arms in
  the main clone; a negative control confirmed lit does fail (exit 2) on a corrupted CHECK.
- **the 4 lit arm-2 failures** — `Generic/bswap.ll`, `Generic/dwarf-source.ll`,
  `Generic/dwarf-md5.ll`, `RISCV/rvv/debug-info-rvv-dbg-value.mir`. All four fail identically with
  the compiler change reverted and rebuilt.

## What was NOT gated — read this before building on the stack

The stack's **own** gates did not run: `capstone/runtime/virtual/run.py` and
`capstone/tests/virtual-runtime-r0/run.py` need a prepared kernel tree matching the booted image,
a cross compiler prefix, a loadable module and a Linux guest. None of that is built on this host.

So the new virtual path — the trusted Linux adapter, virtual libc mappings, the shared revoking
heap, same-mm pthreads, and the eight application profiles — is landed on **regression evidence,
not qualification**. The PR author says so directly in `capstone/runtime/virtual/README.md`:

> "Review integration: checked-in qualification JSON files identify historical binaries by hash.
> They do not qualify this extracted stack on current dev."

The figures those files carry (pthreads 9/9, v2 compatibility 39/39, R1 22/22) are the author's,
from their environment, and were not reproduced here. `run.py --skip-recycling` additionally omits
two long recycling cases — disclosed in its help text and in the qualification JSON, not hidden.

## Two latent hazards worth fixing before anyone uses the adapter

- **The kernel module is built out-of-tree with no `-march` pinning.**
  `capstone/runtime/virtual/build-adapter.sh:11` and `capstone/tests/virtual-runtime-r0/build.sh:14`
  run `make -C "$KERNEL_BUILD" ... M=<dir> modules`, and `module/Makefile` is only
  `obj-m += capstone_vm.o`. That path does not pick up `external.mk`'s `rv64g`, which is the
  recorded shape of an RVC module **hanging the board during `insmod`** (misaligned relocation
  read, no cause-4 handler). Harmless under QEMU, which executes RVC; latent if the adapter is
  ever pointed at the board.
- **`-j16` is hardcoded** in both of those scripts, against this host's `-j12` cap for agent work.
