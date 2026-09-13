# Plan (approved 2026-09-13): settle sw64 on a proven matched pair, then land the collaborator's PRs in dependency order

_Approved in session on 2026-09-13; execution state is tracked in `docs/state/current-next-step.md`._


## Context

**The PRs.** Eleven PRs from the external collaborator, in six groups. All eleven were fetched
(`refs/pr/N`, and the submodule ones inside their submodules), diffed, dry-merged, scanned, and read
— by me and by three reviewers whose claims I re-checked against the primary sources. What is settled:

* **Every diff is clean of names and emails** (0 added lines with either, in all eleven). The
  precommit scan still BLOCKS every range, and in every case 100 % of the hits are the git
  **author/committer identity** — the collaborator's email carries their real name. The same identity
  is already on `origin/dev` **130 times** (13× on the QEMU branch), landed by the lead as plain
  merges ("Merge #10 …", "Merge the external collaborator's speedtest1 bring-up stack (#7, #8, #9)")
  after the identity scan existed (2026-09-07). The scan's own comment (`precommit-scan.sh:50-55`)
  says identity lines are scanned so that a *cherry-pick* cannot smuggle a name; a plain merge of
  commits that are **already on `origin/*`** (every PR head is) leaves the push range
  `<branch> --not --remotes` holding only our merge commit. So the established route is a plain
  merge, scanned over the range the push actually publishes. A cherry-pick makes us the committer
  and **is blocked by design** — which matters for `capstone-qemu #3` below. **The lead confirmed
  (2026-09-13): plain merges, as before.** So every merge below is a merge commit of commits already
  on `origin/*`, and the scan runs over the range the push publishes.
* **Dependency graph, verified by ancestry:** buildroot `#2 ⊂ #3`; llvm `#11 ⊂ #12 ⊂ #13`;
  `#16 ⊂ #17`. Cross-repo: **`#17` models exactly what buildroot `#3` implements**
  (`MONITOR_SPLIT_SLACK (8*1024)`, one region sized `code_len + slack + domreq_data` — read in
  `#3`'s hunk), so `#17` is wrong until `#3`'s module is built and right afterwards; **`#18`'s
  headline numbers need `#2+#3`** (its 160-test configuration dies in the entry glue under the
  code-size fallback — the collaborator's own commit text, confirmed: no `domreq` reader exists in
  any checkout). `#13` and `#18` **conflict in `run-nightly.sh`** (three regions, mechanical).
  `#14` conflicts in `ISSUES.md` only (dev added S-15 above the S-14 heading the PR rewrote).
  `#15`/`#16` are not duplicates — same shape, disjoint files.
* **`capstone-qemu #3` must not be merged as-is.** Two of its four commits are already on
  `c128-qemu-merge` in *superset* form (`c64867389e`, `b59d116983`); the 11 conflict hunks in
  `op_helper.c` are those two, and resolving any of them toward the PR reverts `cabc953e58`
  (`badaddr`, misaligned-store cause 6), the Q-07 cursor advance, `tagwatch_granule` and the S-12
  probe. Net-new content is the `storewatch` (last two commits, ~104 lines, `git apply --check`
  clean on our base). It touches neither `helper_csdelin` (S-15 stays hidden in QEMU) nor any abort
  path (`trapctl` still cannot earn a QEMU pass).
* **`#14` is a codegen change whose lit test does not gate the defect.** I ran the PR's test through
  the *unpatched* `llc`: both `CHECK`s already pass (`sd ra, 432(sp) # 8-byte Folded Spill` is the
  prologue, `stc ra, 176(sp) # 16-byte Folded Spill` is already a capability store at `-O2`). The
  physical-register path the fix changes is only reached at `-O0`, which both domain builds use —
  so every `-O0` gp-free image rebuilt after `#14` has **different bytes at the same addresses**.
  The gate that can see the defect is `capinit-scan.py` over a built image.
* **buildroot `#3` leaves R-33 intact** — its hunk is in `ioctl_create_dom`; `ioctl_create_region`
  (where `capstone_repr_round` runs) is untouched, and the domain block handed to `DOM_CREATE` is
  `(1 << order) * PAGE_SIZE`, representable by construction. SQLite silicon geometry is
  byte-identical declared or undeclared (`tot_size 4194304`). Two latent points for the
  collaborator: the 8 KiB slack is short once `repr_len ≥ 4 MiB` (granule 8 KiB), and the
  header's skew claim holds in one direction only.

**sw64 / sw65.** sw65 was **not a matched pair** for sw64 and its failure is uninterpretable
against sw64's: it changed three things at once — `INTERP_DOMAIN_MTVEC`, the feature set
(sw64's image has rtree compiled in = `SQLITE_FULL=on`; sw65's does not), and the source state
(sw64: worktree at buildroot `22f3faf9`, 2026-09-11 07:09; sw65: `8da1559`/dev, 13 commits
later). Worse, **sw65's image never ran on QEMU at all**: `build-sqlite-silicon.sh`'s own smoke
has no `cma=` plumbing, so for a 128 MiB region arena it died at `C2/mkarena`
("create_region(arena) failed -- above 4 MiB this needs a CMA area", `__EXIT_CODE__1`) and the
build printed `Built …` anyway; `preflight-board-run.sh` checks a QEMU record only for the
**control**, so the image went to the board unvalidated. Its `sqlite3_initialize refused`
(0x5117BAD3) is therefore a property of an unexercised build, not yet a silicon fact. sw64's image
(`23da3b126a304585`) *did* pass QEMU at 128 MiB (the handoff's size-1 row, via
`run-speedtest1-measure.sh:207`'s `cma=${SPEEDTEST1_CMA_MB:-64}M` — note the **64 MiB default**,
which is below the arena; it must be set to 256 explicitly). `run-domain-smoke.py` takes
`--kernel-arg cma=256M` and the guest has 8 GiB, so the 128 MiB configuration is fully runnable
off-board.

**Why the order below.** Two of the merges change exactly what the matched pair must hold
constant: buildroot `#3` rebakes the module the board boots, and `#14` rebuilds the compiler
(`-O0` bytes change). The pair needs today's `.ko`, today's `llc`, and sw64's own image — so the
sw64 work goes **first**, the merges after.

---

## 1. sw64 / sw65 — reproduce off-board, prove the pair is matched, then two boots

1. **Run sw65's exact image on QEMU** (`33399c3629281d88` + host `ec5abac09a602404`,
   `SPEEDTEST1_CMA_MB=256`, size 1). If 0x5117BAD3 reproduces: bisect on QEMU with the existing
   stage selector (`sqlite_capstone_domain.c:1508-1548`, stages 4–10 split `sqlite3_initialize` at
   its own boundaries; runtime override `sel & 0xff`) — no board. If it passes: silicon-only, and
   stages 4–10 go to the board batched in **one** boot (same image, different selectors).
2. **Prove the source state before building the pair.** From the candidate commit (worktree at the
   `speedtest1` bundle head `6128a619`, buildroot `22f3faf9`), rebuild sw64's configuration
   **without** mtvec (`SQLITE_FULL=on`, region arena, `SPEEDTEST1_HEAP=134217728`) and require it to
   reproduce `23da3b126a304585` — or, if a build timestamp is embedded, byte-identical `.text` and
   disassembly. A rebuild that does not reproduce sw64's image means the wrong commit, and the pair
   would carry an unnamed variable — the exact thing that made sw65 uninterpretable. Only then flip
   `INTERP_DOMAIN_MTVEC=1`, on today's `llc` (before step 2.7 rebuilds it). Verify by artifact:
   rtree symbols present, `csrw mtvec` present, hash recorded; QEMU size-1 with `cma=256M` must
   PASS — that pass is the image's licence to be staged.
3. **Two boots, each control-first, watchdog wired** (`ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420`):
   boot A = `k800` + **sw64's original image** `--size 20`; boot B = `k800` + the mtvec pair image
   `--size 20`. Two boots because both images link at the same entry VA and the pair must not differ
   in link address. **Pre-registered known differences from sw64 itself** (written down before the
   boot): monitor `4274268` (RCRE) instead of sw64's `2c49c41` — RCRE sits in the reclaim path,
   gated on UNINIT, and is expected inert on the SHA5→SHA6 path; the R-33 rounding module — 2²⁷
   rounds to itself, no `not representable` line expected. So boot A is a redraw *modulo the
   monitor*, not an exact one. Pre-registered readings: `mcause/mepc` returned = the fault is named;
   `SHA5` with no `G/enter` **with mtvec installed** = not a trappable fault, an R-16-class per-image
   entry stall, retrying is futile; `H/return` = sw64 was a one-off draw. The `--size 100` decision
   waits on these readings.
4. **Close the gate hole** (after step 2.5, since `#11`/`#13` touch `build-sqlite-silicon.sh`):
   the build's smoke passes `--kernel-arg cma=<2×arena>M` whenever `SPEEDTEST1_ARENA_SIZE > 4 MiB`,
   and a smoke failure makes the build **exit non-zero without printing `Built`**. Positive
   control: rerun sw65's exact build — the smoke must now reach `SQ: G/enter` instead of
   `C2/mkarena`. Add to `preflight-board-run.sh` a check that every staged SQLite image has a QEMU
   smoke pass **by hash at the intended arena size**, BLOCKED otherwise (the control-only check let
   `33399c3629281d88` through).

## 2. Land the PRs — in this order, one plain merge each

Per merge: `git fetch origin` first (so the PR commits sit on `origin/*`); merge commit titled
`Merge <repo> PR #N from the external collaborator: <subject>`; then
`bash <repo>/capstone/tests/precommit-scan.sh --range '<branch> --not --remotes'`
by absolute path, gated on its exit status (`scan && push`); push at each stable point (`dev`,
`capstone-bootstrap`, `c128-qemu-merge` are on the allowlist). Submodule first, parent after.
`git show` (not `--stat`) on every merge before pushing.

| # | PR | Action | Before it lands |
|---|---|---|---|
| 1 | llvm **#15** toolchain-fresh | merge | run `toolchain-fresh.py` on dev *before* (returns early, rc=2 at the drift) and *after* (reaches the staleness check) — the pass has to be shown to mean something |
| 2 | llvm **#16** domdata-budget model | merge, then one follow-up commit of ours fixing the now-false print at `domdata-budget.py:288` ("falls back to max(2*code_len, 512K)") | positive control: on sw65's image (`code_len 1483672`) it must say **order 10, fits** — the module allocated order 10 (sw65 entered) where the old model said order 11 / DOES NOT FIT |
| 3 | buildroot **#2 → #3** | merge `#3` (fast-forward on `capstone-bootstrap`), push, then **fast-forward `caplifive-system/sw/buildroot`** — these two checkouts share ONE remote; the `caplifive-system-dev` copies are on a different line and are left alone — and verify the two copies of `module/capstone.c`, `libcapstone.c`, `include/capstone.h` byte-identical by hash; rebake `modcapstone-rebuild, linux-rebuild, opensbi-rebuild` | `strings` the built `.ko` for BOTH "not representable" (R-33) and "domain declares dom_data" (#3) — the FPGA builds the `caplifive-system` copy, so a merge in one checkout changes no board image |
| 4 | llvm **#17** | merge **only after 3** and its `.ko` exists; follow-up commit of ours deleting the "read by NOBODY" block it leaves behind | positive control: a 3 MiB-stack declaration on the SQLite image must now report **DOES NOT FIT (order 11)** — the case the old two-region model passed and the module cannot allocate |
| 5 | llvm **#11 → #12 → #13** PostgreSQL | merge `#13` (contains all three; `#12` alone leaves a README documenting files that only `#13` adds) | rebuild `sqlite_silicon.dom` before/after `#11`'s `sublet.h` move — `.text` and disassembly identical (hash, unless a timestamp is embedded); run `run-pg-gate.sh` and `run-pg-sublet-gate.sh` once each under the QEMU lock (serialised) and keep their result lines |
| 6 | llvm **#18** MicroPython | merge after 3 and 5; resolve `run-nightly.sh` by hand (help line names both ports, both `_DIR` blocks, both suites) | run `run-micropython-gate.sh` once (QEMU): the only way the uncited `PASS=100 FAIL=60` claim becomes reproducible here — record the actual numbers in the merge message, whatever they are |
| 7 | llvm **#14** S-14 codegen | **last**, kept in this session (codegen): resolve `ISSUES.md` (keep S-15, take the new S-14 heading but word it "fix landed, verified below"); make the lit test discriminate — add an `-O0` RUN line and **prove it FAILS on the unpatched `llc` before rebuilding**; if `-O0` does not reproduce either, delete the test's coverage claim (`:28-31`) and say `capinit-scan.py` is the gate | rebuild toolchain (never during a suite); lit + corpus-runner QEMU suites; `capinit-scan.py` positive control — the OLD SQLite image must trip it, the new one must scan clean; claim-auditor on "S-14 FIXED" before the heading says so; then re-run the QEMU size-1 bridge pair (icount ratio ≈ 1.270) since every `-O0` image now differs in bytes |
| 8 | capstone-qemu **#3** | **not as-is.** Ask the collaborator (note for the lead under `/tmp/capstone/`, no names) to rebase onto `c128-qemu-merge` dropping the two already-landed commits, then plain-merge. Fallback if the lead wants it now: merge with all 11 hunks resolved toward ours plus the storewatch additions | either way, PROVE no revert: `git diff <merge>^1 <merge> -- op_helper.c` must equal `git diff refs/pr/3~2 refs/pr/3` modulo context; rebuild QEMU; QEMU suites; storewatch positive control — `CAPSTONE_STOREWATCH` on a known write prints it |

Accepting `#11`'s new top-level `capstone/sublet/` is a layout precedent (argued in
`repo-layout.md:26-37`); I take it as yes unless the lead objects at review.

## 3. Hand-offs (to the collaborator through the lead; a note under `/tmp/capstone/`, no names)

* `capstone-qemu #3`: rebase onto `c128-qemu-merge`, drop the two already-landed commits.
* buildroot `#3`: compute the split slack from the monitor's granule instead of the 8 KiB constant;
  the header's skew claim holds only loader-new/module-old.
* llvm `#14`: the lit test does not discriminate at `-O2`; needs `-O0` or an honest claim.
* llvm `#18`: the 160-test result needs its log or a "measured with #2+#3" qualifier until
  reproduced here (step 2.6 does that).

## Verification

* Step 1.2's reproduction of `23da3b126a304585` (or identical `.text`/disassembly) is the
  precondition for spending boot B at all; without it the pair is not a pair.
* Every merge: scan by absolute path over the range the push publishes, gated on exit status;
  `git show` read in full; each new nightly suite run once and seen to produce a result line
  (a suite that has never run is not a gate).
* Every "positive control" in the tables above is run **before** the corresponding pass is
  believed: `#15` before/after, `#16`/`#17` on the sw65 image and the 3 MiB-stack case,
  `capinit-scan.py` on the old image, the storewatch on a known write, the fixed build smoke on
  sw65's configuration.
* After buildroot `#3`: `.ko` content by `strings` for both markers; the two shared-remote copies
  byte-identical by hash.
* After `#14`: lit green, QEMU suites green, the QEMU size-1 bridge pair ratio re-measured.
* Board: control `retval=4` or the boot is void; readings judged on `mcause/mepc`, not on
  completion; results cited by image hash; the pre-registered monitor/module differences written
  into the driver header before the boot.
* No `Co-Authored-By:`; no names anywhere, including the merge subjects and the `/tmp` note's
  filename.

## Documents to update

`docs/ref/fpga-silicon-measurements-for-paper.md` §7n (sw65 was not a matched pair; its image
never ran on QEMU; the feature-set and source differences named), `docs/ref/ISSUES.md` (S-14
heading only after step 2.7's verification; M-1 unchanged), `docs/state/current-state.md` and
`current-next-step.md` (the PR landing table with what landed and what was held, the sw64 status,
and the still-open R-33 items carried forward explicitly: the `--pool` positive control for the
rounding log and the bottom-truncation arm), a dated `docs/history/` note for the review itself
(per-PR verdicts and the evidence, no names), this plan committed to `capstone/docs/plans/` on a
branch off `dev`, and the collaborator hand-off note under `/tmp/capstone/` (never committed).
