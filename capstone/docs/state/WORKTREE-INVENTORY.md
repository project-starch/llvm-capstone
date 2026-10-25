# Worktree inventory — `llvm-capstone`, 2026-10-13

56 worktrees are registered on this machine and 12 of them had uncommitted
changes. This file is the audit; **nothing has been removed.** Removing a
worktree is cheap to do and expensive to get wrong, so what is recorded here is
the evidence a removal would rest on, not the removal.

## The method, and the one thing it cannot tell you

Each worktree's HEAD was tested for ancestry in `origin/dev`
(`baa1ab2faded`, fetched 2026-10-13). That test is sound in ONE direction
only:

- **ancestor ⇒ the worktree holds no commit of its own.** Nothing is lost by
  removing it, once its uncommitted changes are secured.
- **not an ancestor ⇏ unlanded.** Lanes land on shared branches *squashed*
  (CLAUDE.md, "Squash when landing"), so a landed lane branch keeps commits
  that are not ancestors of `dev` and shows here as "not an ancestor" with a
  commit count. **The count below is not a measure of unlanded work**, and a
  clean worktree is not evidence of a merge either. Deciding the 45 needs the
  per-lane question "did this land, squashed, as which commit" -- which is a
  judgement per lane, not a command.

## Uncommitted changes: secured first

Every dirty worktree's uncommitted state was archived on 2026-10-13, before
anything else, to `~/worktree-safekeeping-2026-10-13/<worktree>/` -- outside
the repository and outside `/tmp`:

- `state.txt` -- branch, HEAD and the full `git status --porcelain`
- `tracked.diff` -- `git diff HEAD`, staged and unstaged together
- `untracked.tar.gz` -- the untracked, non-ignored files as found
- `MANIFEST.sha256` at the top -- every file in the archive

3.8 MB in total for 12 worktrees.

### `llvm-capstone-heap-sublet`: 171,657 staged deletions, and no work

This is the one that looked alarming and is not. Its index marks all 171,657
tracked files deleted while only `capstone/` (2,648 files) is on disk, so its
`git diff HEAD` was 41.5 million deleted lines. Every file on disk was hashed
against the worktree's HEAD (`f5ba3d3626cb`): **2,646 of 2,648 are
byte-identical**, one differs only in line endings (a results `matrix.tsv`,
CRLF against LF, same 97 rows), and one is a submodule gitdir pointer. The
deletions are an index artifact, not a change. Its branch
`runtime/virtual-heap-sublet-lifetimes` sits at the same commit as
`ports/nested-controls-extended`, so the commit outlives the worktree. The
archive keeps the listing and the files; the 1.9 GB diff was not worth keeping
and `WHY-NO-DIFF.txt` beside it says so.

## Removable: HEAD is an ancestor of `dev` (11)

Nothing in these exists only here. Each is a candidate for
`git worktree remove`, in this order, after the archive above is accepted.

| worktree | branch | head | uncommitted |
|---|---|---|---|
| `~/llvm-capstone-delegation-spawn` | `delegation-ports` | `8546ea08a5d9` | 1 |
| `~/llvm-capstone-mallocbench-virtual` | `malloc-bench-virtual` | `adc891bb920c` | 1 |
| `~/llvm-capstone-ports` | `ports/14-poisoncap-postgres` | `5297b423822c` | 1 |
| `~/llvm-capstone-sublet-lifetimes` | `isa/sublet-lifetimes` | `b02ddf9043cc` | -- |
| `~/llvm-capstone-virtual-addressing` | `virtual-addressing` | `6b5779a09072` | 2 |
| `~/llvm-capstone-virtual-guide` | `virtual-capstone-guide` | `0d454b2469e6` | -- |
| `~/llvm-capstone-virtual-musl-local` | `virtual-musl-local` | `adc891bb920c` | -- |
| `~/llvm-capstone-virtual-musl-malloc` | `virtual-musl-malloc` | `f2c70e991714` | 20 |
| `~/llvm-capstone-virtual-node-growth` | `review/virtual-node-growth` | `f2c70e991714` | -- |
| `~/llvm-capstone-virtual-ports-review` | `virtual-memcached-server` | `1354aee956cd` | -- |
| `~/llvm-capstone-virtual-review` | `virtual-pthreads` | `518a5b805aa7` | -- |

Three of them carry untracked files that are already on `dev` by path
(`llvm-capstone-virtual-musl-malloc` is the clearest: all five of its
untracked files are present on `dev`), which is consistent with the work
having landed and the worktree not having been cleaned up afterwards.

## Not an ancestor of `dev` (45)

Read the caveat above before reading this table: the count after "no" in the
audit is commits not reachable from `dev`, which for a squash-landed lane is
most of the lane. These are listed so the set is known, not ranked for
removal.

| worktree | branch | head | uncommitted |
|---|---|---|---|
| `~/llvm-capstone` | `sqlite/5-sealed-callback` | `3a1668133bd3` | 2 |
| `~/llvm-capstone-capinit-alias` | `compiler/cap-init-alias` | `fd82fef2a70d` | -- |
| `~/llvm-capstone-compiler-rebase` | `lane/compiler-stack-20260929` | `68c75ed3e1ad` | -- |
| `~/llvm-capstone-controls` | `ports/nested-lifetime-controls` | `88d0960bf9ae` | -- |
| `~/llvm-capstone-controls-ext` | `ports/nested-controls-extended` | `f5ba3d3626cb` | -- |
| `~/llvm-capstone-cpydemo` | `demo/cpython-no0006` | `a1821f6d2c0c` | 2 |
| `~/llvm-capstone-encoding` | `delegation-memory-encoding` | `cd0e38f011a6` | -- |
| `~/llvm-capstone-fd-rows` | `delegation-fd-rows` | `461078a55081` | -- |
| `~/llvm-capstone-feasibility` | `trusted-linux-feasibility-gate` | `3638abe58a4c` | -- |
| `~/llvm-capstone-ffmpeg-app-sublet` | `ffmpeg/app-sublet-pools` | `8419196afe2b` | -- |
| `~/llvm-capstone-ffmpeg-decoders` | `ffmpeg/wider-decoders` | `5d94bf5810ef` | -- |
| `~/llvm-capstone-heap-sublet` | `runtime/virtual-heap-sublet-lifetimes` | `f5ba3d3626cb` | 171658 |
| `~/llvm-capstone-intcap-type` | `compiler/sroa-keep-capability-whole` | `7d01722aab88` | 2 |
| `~/llvm-capstone-malloc-quarantine` | `experiments/cheribsd-malloc-quarantine` | `032a1aed840c` | -- |
| `~/llvm-capstone-mallocbench-c` | `virtual-heap-no-temporal` | `5343580810a0` | -- |
| `~/llvm-capstone-memcached-lifetimes` | `nested/memcached-lifetimes` | `358ebbacdcdb` | -- |
| `~/llvm-capstone-memory` | `delegation-memory` | `d4fdce9bf58a` | -- |
| `~/llvm-capstone-model` | `lifetimes/1-one-bit-per-identity` | `259f6c8d8e65` | -- |
| `~/llvm-capstone-nested-rebased` | `ports/nested-allocators-rebased` | `2694a68704ae` | 2 |
| `~/llvm-capstone-no-sublet-heap` | `cleanup/no-sublet-heap` | `3b5b5dcb3841` | -- |
| `~/llvm-capstone-perl` | `perl/1-musl-domain` | `92e3ca824ba3` | 8 |
| `~/llvm-capstone-perl-corpus` | `ffmpeg/arm-design` | `59fb8e105ed6` | -- |
| `~/llvm-capstone-pg175-terms` | `corpus/postgres-17-5-terms` | `6d707b7bf34a` | -- |
| `~/llvm-capstone-physical-tags` | `trusted-linux-m1-context-optin` | `ab3352447025` | -- |
| `~/llvm-capstone-qemu-stage1` | `delegation-memory-qemu` | `b14b66e8c572` | -- |
| `~/llvm-capstone-reuse-rep` | `ports/reuse-repeats` | `b3bf704d9bc7` | -- |
| `~/llvm-capstone-revnode-perf` | `perf/revnode-cache` | `eed26ea733cb` | -- |
| `~/llvm-capstone-runtime-plan` | `virtual-capstone-runtime-plan` | `bd13d9348b75` | -- |
| `~/llvm-capstone-runtime-r0` | `virtual-capstone-review-plan` | `e03131496304` | -- |
| `~/llvm-capstone-same-answers` | `ports/protected-same-answers` | `55f34f5e2e6c` | -- |
| `~/llvm-capstone-syscall-bounds` | `trusted-linux-syscall-bounds` | `923cc8c235ae` | -- |
| `~/llvm-capstone-tags` | `tags/1-slot-identity` | `1c15b651191a` | -- |
| `~/llvm-capstone-threads` | `delegation-threads` | `92d21b84324b` | -- |
| `~/llvm-capstone-trusted-linux` | `memory-trusted-linux` | `6ba9673ab2dc` | -- |
| `~/llvm-capstone-udp-recv` | `trusted-linux-m1-boundary` | `bc43ffd47ad6` | -- |
| `~/llvm-capstone-user-resume` | `plan/user-mode-resume` | `fefefe9110df` | -- |
| `~/llvm-capstone-v0-removal` | `delegation-v0-removal` | `cabfaa34b3a9` | -- |
| `~/llvm-capstone-virtual-corpora` | `virtual-capstone-bug-corpora` | `1ee1cc40316f` | -- |
| `~/llvm-capstone-virtual-integration` | `virtual-capstone-integration` | `caa19c48ec34` | -- |
| `~/llvm-capstone-virtual-limits` | `virtual-vm-sampling` | `ead321edacb8` | -- |
| `~/llvm-capstone-virtual-musl-policy` | `virtual-musl-policy` | `fdd9aa975337` | -- |
| `~/llvm-capstone-virtual-prototype` | `virtual-capstone-prototype` | `55708215b540` | -- |
| `~/llvm-capstone-ws-cands` | `wireshark/live-candidates` | `dfb2142991c3` | -- |
| `/tmp/capstone-qemu-separate-compilation-plan` | `plan/qemu-separate-compilation` | `2ed52d3f0bc4` | -- |
| `/tmp/capstone/pg-policy-isolation` | `(detached)` | `3766a15631ad` | 5 |

## What is owed

1. The 11 above can go once the archive is accepted. That is ~20% of the
   worktrees for no loss.
2. For the 45, each lane's landing has to be established per lane -- its
   squashed commit on `dev`, or a `backup/*` tag as
   [`BRANCH-INVENTORY.md`](BRANCH-INVENTORY.md) does for `capstone-ariane`.
   A worktree whose lane landed and whose branch is tagged can then go too.
3. `/tmp/capstone/pg-policy-isolation` is a detached worktree under `/tmp`
   with five uncommitted changes, including a new PoisonCap patch. `/tmp` is
   not storage; its changes are in the archive, and the lane it belongs to
   should claim them.
