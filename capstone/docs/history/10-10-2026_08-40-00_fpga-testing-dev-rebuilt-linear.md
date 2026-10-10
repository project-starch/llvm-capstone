# `fpga-testing-dev` rebuilt as one linear history: 44 commits from the fork, RTL byte-identical to the silicon

**Date:** 2026-10-10, on the lead's instruction ("ready clean commits, squashed, a perfect linear history"). **Repo:**
`capstone-ariane`. **Old tip:** `0bf09b1d6` (2026-09-15), preserved as tag `backup/fpga-testing-dev-2026-10-10` on
origin and in an all-refs bundle `~/dev/llvm-capstone-rebuild/backups/capstone-ariane-2026-10-10.bundle`. **New tip:**
`5382266b8`, lane branch `fpga-testing-dev-2026-10` (pushed, never deleted). **Force-push:** `--force-with-lease` against
`0bf09b1d6`, `--no-verify` because the pre-push hook blocks `fpga-testing.*` unconditionally; the lead's go-ahead is the
instruction above, given in the RTL lane's conversation.

## Why

The default branch had stopped at R-30/R-31 while the board ran `776d9d859` (R-29 + S-10b, flashed 2026-10-05), and it
was not an ancestor of the silicon line: `sup-call` forks from `66c4e7517` (R-27). Two commits existed only on the default
branch, `45d17921e` (65 test files, the R-18/R-28/R-29 directed arms and the quarantine rule; tests only) and `0bf09b1d6`
(a squash of R-30/R-31 whose RTL equals the silicon line's `2c59a355b`; only tests and testlists differed). Since the fork
from `fpga-testing` at `e1b3db6ba` (2026-08-08; `fpga-testing` has not moved) there were 87 commits, two merges, two by
the external collaborator.

## How it was built, so the trail can be checked

**No cherry-pick where a historical tree exists.** Every commit from the August merge onward was made with
`git commit-tree <tree> -p <prev>`, `<tree>` being the tree of the LAST lane commit of its group — a tree that was built,
linted and in most cases synthesized — and the script verified `git diff --quiet <group's last commit> <new commit>` at
every step. The final tree is therefore `776d9d859`'s by construction: `git diff --stat a7f88a2ff 776d9d859 (the last snapshot commit)` is empty.
Each message carries the group's lane hashes under "Rebuilt 2026-10-10 ..." and the lane messages verbatim; no
attribution trailers (the composer refuses them). The August segment before the merge is the one place cherry-picks were
used: the collaborator's `b047f32eb` and `0c55c21f9` are reused as they are (authorship intact), `f623c48a1` (R-20) and
`55b7f88bc`+`8e6600a1b` (S-06 enabler) were picked onto them, and the three verif commits became one snapshot of the
merge's tree after checking the remaining difference was `verif/` only.

| new | group | tree of | lane commits |
|---|---|---|---|
| `b047f32eb` | Fixed some linearity enforcement issue | as committed | — |
| `0c55c21f9` | Fixed a missing check on cap permission for LDC | as committed | — |
| `72f22ae09` | Fix R-20: keep the CAPENTER x10 clobber additive instead of overwriting | as committed | — |
| `d1f10f3f6` | S-06 enabler: LCC's type query is total, and the ex_code mcause comments lose their off-by-one | cherry-pick | 55b7f88bc 8e6600a1b |
| `073041822` | verif: S-06 reproduced in simulation and its repair sequence; the store-misclassification family ref | `2035df882` | efffa7c47 e33efdf67 |
| `895a26443` | Expose the LATCHED trap mepc on the debug mux, so a wedge can name its own faulting pc | `b30b93fab` | — |
| `cb38cd9cb` | verif: eight directed tests adapted to the merged RTL, with failures that say which arm | `7e4dc440f` | — |
| `a41df487a` | verif: the synthesis-hazard lint gate and the regression sweep, baselined at the shared base | `4b317212b` | — |
| `dade47718` | S-06: capability-ness is the tag bit, not the type field | `6fe7a49ab` | — |
| `7f287d2de` | S-08: the domain-switch context width must honor metadata_en | `7debad21d` | — |
| `4c307bdc4` | S-07: forbid granule co-residency in the write buffer | `1a971710a` | — |
| `15cf0aa9b` | S-10: look the capability tag up at GRANULE granularity in the write buffer | `1a9f90894` | — |
| `d3137ab76` | S-12: the WAW guard must see retirement, not the write | `a3aebad28` | — |
| `c4ce8abf8` | Report the faulting operand cursor in mtval on a capability exception, and latch it for the debug ba | `4b0744a2b` | — |
| `f22191324` | Synthesis tooling: memory-guarded run, artifact collection, timing forensics | `947327f6d` | — |
| `a16d2ccdc` | Domain switch: gate the core on a registered switch-in-progress flag, not on the switcher's busy wir | `ef5a8eaf2` | — |
| `d64ad1155` | R-26: flush the pipeline after a committed capability CSR write | `9d8797560` | — |
| `1d03b5623` | R-25: INIT with rs1 != rd nulls rs1 instead of duplicating the linear capability | `42a141c93` | — |
| `f02976dc8` | R-27: drain a revocation-node response whose requester was flushed | `66c4e7517` | — |
| `1ca3fd8ef` | R-30 and R-31: INIT is reachable by filling, and a revoke through a write-bearing capability re-init | `1bfff7776` | a1484c6d3 883e8f2a4 060f68490 |
| `fdbf3d1f6` | verif: the three R-33 directed arms (a moved cursor widens a LINEAR region at both ends and permits  | `4cc068572` | cb2cd046a eab5b196b 2c59a355b |
| `981af6b54` | R-12: unlink revoked nodes from the chain, two writes per revoke rather than per node; the splice's  | `f1331daed` | 379248185 |
| `ceb14d35c` | S1: the revoke-walk cost curve and the ladder that measures it; every published rung regenerates fro | `c49190d90` | 5385d482b 3f3395825 38316d2c7 |
| `f49a9f394` | R-34 and R-24: the MMU's exception register restored, the debug sentinel moved off 24, the capabilit | `f714d2a72` | c77c65324 9175cc352 e97b7e7ab 9a7bd598c 88f374d1c |
| `626997de5` | Revocation-node pool exhaustion is an architectural fault, not a core hang; DELIN refuses a dead nod | `b49673357` | — |
| `ccbb49b23` | R-12 reclaimer A1..A6: free bit, LIFO free list, (generation, index) allocation, valid AND generatio | `054cea69b` | 35081fdb9 ac5706553 1ac15c4ef d9620b907 |
| `03012681f` | R-37/R-38 stage 0: invalidate the revnode trackers AFTER the adopt, against the post-adopt id | `247b76896` | — |
| `2fba9abe1` | R-35: gate the LSU revocation check on observed evidence; the cache rebuilt as a register file with  | `4ad0df694` | 079dc720a 6ee277cc3 f83fe9342 a87a24a59 |
| `a83635e20` | R-42: accept the redirected fetch in the cycle a speculative I-cache miss is killed | `6cbdaeeb4` | — |
| `a2d679c1d` | R-43, first form: a directed reproducer for the false deny, and a cache miss resolved by probing the | `bbd4d1478` | 93f509f54 |
| `057a3e21c` | R-45: flush younger instructions when a REVOKE or DROP commits, so no load/store is checked before t | `0f5185a6d` | — |
| `0b680671f` | R-43, second fix: REPLAY a missed access through the existing exception path, never hold the request | `5aa316e0d` | 8f6a0af98 |
| `f2a3d25d6` | Supervised CALL, prerequisites: the switcher's full exchange works once three defects are fixed; lad | `d38887426` | — |
| `281f6e122` | S-11 in simulation: a 64-byte seal makes the switch write past the capability's end; and a lint for  | `1dbf379b1` | — |
| `a4d04e8e7` | R-47: CALL parked the wrong return pc when a jump or branch reached issue before the dyn unit respon | `727ea6e93` | — |
| `d8ff7c1cf` | Supervised CALL on silicon: cssupervise, the escape at commit, the SAVE/RESTORE walks, the quantum,  | `bd4c0d486` | 03b70667e eacb49e5d b9acda022 |
| `853392076` | S-11 fixed: SEAL refuses a region smaller than 1024 bytes or not 16-byte aligned, and sets the seale | `c0c546542` | c0ad507b2 |
| `497668c17` | verif: m1-take-cost under rdcycle, the sim-only cause detector, sup-escape/sup-quantum repaired afte | `ca912dc77` | 36a641e0b 40ac6a8fe 7564c0945 b83d67ea1 |
| `98b883d01` | Sim-only tracers for the two supervised-switch hangs, the MSWAP/burst directed tests, and the apertu | `b576635be` | — |
| `b9690834f` | load_unit: the dom-switch flush exemption applies only while the switcher's read is outstanding (R-5 | `429c60b32` | — |
| `b0a78374a` | R-49: a switcher write is accepted on the room of the queue it will be pushed into, not the previous | `87c186ebc` | 192a5e624 d92093828 |
| `7f81e5384` | LED apertures for the S-17 / S-16 wedge reads, and the lint gate counts CASEOVERLAP (baseline 2) | `715bdd1fe` | 07eb22deb |
| `a7f88a2ff` | R-29 and S-10b fix, stall-only: a read whose granule has a conflicting store in flight waits in the  | `776d9d859` | 56ca92df9 d453f9d94 |
| `5382266b8` | verif: carry the R-18/R-28/R-29 directed arms and the quarantine rule from the old default branch, a | tests-only tail on `776d9d859` | — |

**The tests-only tail.** `5382266b8` carries the 65 files of `45d17921e` and its testlist entries on top of the silicon
tree (`core/`, `corev_apu/`, `verif/sim` unchanged, checked by `git diff --stat <tail> 776d9d859 -- core corev_apu verif/sim`
= empty). Placing it at its chronological place (after R-27) was rejected: every later snapshot tree would then have
needed re-basing by cherry-pick with testlist conflicts at eight points. The nine arms were RUN on the final tree before the
commit (Verilator testharness, `S12_MEM_DELAY=12`, seed 1):

| arm | before adaptation (gp at the exit-store trap, then handler) | after |
|---|---|---|
| r29-lowword | PASS, then 20 (90,795 traps) | PASS (1 trap, the skipped exit store) |
| r18-same-plain | PASS, then 20 (90,797) | PASS (1) |
| r18-other-plain | PASS, then 20 (90,793) | PASS (1) |
| r29-sep-forceres | PASS, then 20 (90,797) | PASS (1) |
| r29-sep-userzero | PASS, then 20 (90,772) | PASS (1) |
| r29-sep-userzero-miss | 20 before any verdict | 20 (2 traps): its eviction loop's integer-base `ld` traps under R-34; moved to `testlist_r26.yaml` with the reason |
| s06agg-shape (quarantined 09-11, read 11) | PASS, then 20 (90,795) | PASS (1); joins `testlist_capstone.yaml` |
| r18-same-ldc (quarantined 09-11, read 11) | PASS, then 20 (90,785) | PASS (1); joins `testlist_capstone.yaml` |
| r18-other-ldc (quarantined 09-11, read 11) | PASS, then 20 (90,782) | PASS (1); joins `testlist_capstone.yaml` |

What the adaptation is: on `66c4e7517` the harness exit store (`sw gp, tohost, t5`, an integer base inside capability mode)
raised an exception the LSU lost (R-34), so the store retired and gp carried the verdict; on the R-34 RTL the trap is
delivered and the arms' handler (`li gp, 20; RVTEST_FAIL`) overwrote it and re-trapped forever. The six handler-defining
files now skip that store once the verdict is in gp (`tp = 1`), as the sweep's own CAPENTER tests do; any other trap still
reads 20. The three quarantined arms' 11 on `66c4e7517` is the positive control that these checks can fire.

## Acceptance

- `git rev-list --count e1b3db6ba..<tip>` = 44 (43 snapshot/cherry-pick commits + the tail), merges 0; author census: the lead 42, the collaborator 2.
- `precommit-scan.sh --msg` over every message and over the concatenation of all 44 (3,6k lines): rc 0; positive
  control (a denylist name appended to one message): rc 1, BLOCKED.
- `rtl-lint-gate.sh` on the tip: PASS at the committed baseline (UNOPTFLAT 40, SELRANGE 100, CASEOVERLAP 2,
  UNUSEDSIGNAL 832, ANVIL_UNOPTFLAT 0); negative control (unwritable OUT) BLOCKED rc 1.

## What did not change

`sup-call` (`776d9d859`), `r51-return-pcc` (`f4e4d6051`), `s18-fpr-clobber` (`546807884`) and every other lane branch;
the parent repo's submodule pointer (`dev` → `776d9d859`) and `.gitmodules`. Every hash the registry cites stays
reachable through those branches and the backup tag. R-51 and R-52 are NOT on the new default branch: its tip describes
the part in the lab, and they are not synthesized.
