# Eleven collaborator PRs: review, verdicts and landing order (2026-09-13)

Six groups from the external collaborator, reviewed by reading every diff, dry-merging every head
against its target, scanning every range, and re-checking three reviewers' claims against the primary
sources. What follows is the evidence, not the summary; the summary is in `current-state.md`.

## The scan verdict, and why it is not a blocker

Every one of the eleven ranges BLOCKS under `precommit-scan.sh --range`, and in every case **100 % of
the hits are the git author/committer identity lines** — the collaborator's email carries their real
name. **Zero added lines in any diff carry a name or an email** (checked with the denylist itself,
masked). The same identity is already on `origin/dev` 130 times, landed by the lead as plain merges
after the identity scan existed (2026-09-07). The scan's own comment says identity lines are scanned so
that a *cherry-pick* cannot smuggle a name; a plain merge of commits already on `origin/*` (every PR
head is) leaves the push range `<branch> --not --remotes` holding only our merge commit. The lead
confirmed: plain merges, as before. Consequence for `capstone-qemu #3`: a cherry-pick of its useful
commits is blocked by design; it needs a rebase from the collaborator.

## Dependency graph (verified by ancestry, not by the PR descriptions)

`buildroot #2 ⊂ #3`; `llvm #11 ⊂ #12 ⊂ #13`; `#16 ⊂ #17`. `#17` models exactly what `buildroot #3`
implements (`MONITOR_SPLIT_SLACK (8*1024)`, one region sized `code_len + slack + domreq_data`) — read in
`#3`'s hunk, not inferred — so it is wrong before `#3`'s module exists and right after. `#18`'s headline
result needs `#2+#3` (no `.capstone_domreq` reader exists in any checkout; the collaborator's own commit
text says the 160-test configuration dies in its entry glue without it). `#13` and `#18` conflict in
`run-nightly.sh` (three regions, mechanical). `#14` conflicts in `ISSUES.md` only. `#15`/`#16` are not
duplicates: same shape, disjoint files.

## Per-PR verdicts

| PR | verdict | the evidence that decided it |
|---|---|---|
| llvm #15 toolchain-fresh | **landed** `376486fca526` | before the merge `toolchain-fresh.py` returned early at the drift (rc=2, staleness never checked); after it the check runs and reports the toolchain STALE — the `opt`/`llvm-symbolizer` targets it adds were never built |
| llvm #16 domdata-budget | **landed** `0d9a5091b48e` + `47f3ba497638` (the one print the PR left describing the removed rule) | the module's arithmetic is `code_len + max(code_len, DOMAIN_DATA_SIZE)`, no `DOMAIN_MIN_FREE`; on sw65's image the model now says order 10 / fits, and the module allocated order 10 (the domain entered) |
| buildroot #2 → #3 | held for the board-free window | leaves `ioctl_create_region` and R-33's rounding untouched; the domain block is `(1 << order) * PAGE_SIZE`, representable by construction; SQLite geometry byte-identical declared or not. Two latent points for the collaborator: the 8 KiB slack is short once the granule is 8 KiB; the header's skew claim holds one way only. Must be mirrored into `caplifive-system/sw/buildroot` (same remote) — the FPGA builds that copy |
| llvm #17 | held until #3's `.ko` exists | correct only against #3's module; keeps a "read by NOBODY" block its own flag flip contradicts |
| llvm #11 → #12 → #13 PostgreSQL | **landed** `36f9ca8767d4` (all three: #12's README documents files only #13 adds) | `#11`'s `sublet.h` move is byte-neutral — the Sublet SQLite image is hash-identical before and after (`ceeded2533a74bce`); nothing vendored (pinned 17.5 tarball fetched outside the repo); both gates PASS under QEMU after the merge (`run-pg-gate.sh`: replays in a domain and every freed object held what was written; `run-pg-sublet-gate.sh`: nineteen level-below claims hold, a teardown is one revocation) |
| llvm #18 MicroPython | held until #3 | nothing vendored (pinned clone + 18 in-repo patches, loud on a stale patch); its `PASS=100 FAIL=60` at 160 tests has no log and needs #2+#3 to reproduce |
| llvm #14 S-14 codegen | held, last | the fix gates the `ra` truncation on `FrameSetup`/`FrameDestroy`; the CSR paths pass those flags, so the prologue is unchanged. But the lit test does not gate the defect: on the unpatched `llc` both CHECKs already pass at -O2, and -O0 emits no `ra` spill for that function at all; `capinit-scan.py` over a built image is the gate. Every -O0 gp-free image rebuilt after it changes bytes |
| capstone-qemu #3 | held for a rebase | two of its four commits are already on `c128-qemu-merge` in superset form (`c64867389e`, `b59d116983`); the 11 conflict hunks are those two, and resolving toward the PR reverts `cabc953e58`, the Q-07 cursor advance, `tagwatch_granule` and the S-12 probe. Net-new is the storewatch (~104 lines), which applies cleanly alone. Touches neither `helper_csdelin` nor any abort path |

## Two things this review changed about how a PR is checked here

* **Byte-identity is checked on the build that the change can reach.** `#11` touches only the Sublet
  include path, so the default SQLite build could not have shown a difference; the Sublet build did.
* **A PR's own test is run on the unpatched compiler first.** `#14`'s test passing before the fix is
  what showed it could not fail after it.

Hand-off notes for the collaborator (qemu #3 rebase, the slack constant, the #14 test, the #18 log)
are under `/tmp/capstone/`, never committed.
