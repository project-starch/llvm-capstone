# Proposal: retire `caplifive-system`, make `caplifive-system-dev` the submodule

**Status: PROPOSAL (2026-09-04), for review. Nothing is implemented.** It changes a submodule URL,
which every lane and every fresh clone is affected by, so it is not a change to make unilaterally.

Raised by the project lead: **`caplifive-system-dev` is a full replacement for
`caplifive-system`, and `caplifive-system` should be archived and no longer used.**

> **RESOLVED 2026-09-04 — this section is obsolete, and the framing under it was wrong.**
> The project lead pushed back on it correctly: *why push `caplifive-system` at all, when the
> whole point is to replace it?*
>
> Two of the three levels were pushed (`caplifive-opensbi` `769939a..460f6e4`, buildroot
> `f11bf69..80e1dcf`). Checked afterwards what the outer 4 commits still carry, and the answer is
> **nothing that is not now public**: each one is a `sw/buildroot` gitlink bump, and all four
> targets — `17e4fb609`, `03f3f2832`, `f11bf691b`, `eae3fff72` — are on
> `origin/capstone-bootstrap-dts-65536`. The monitor-to-buildroot correspondence they were
> valuable for now exists on a remote independently of them.
>
> So `caplifive-system` **does not need to be pushed**, and its 403 is no longer a blocker for
> anything. Their history is in `/extra/alexey/caplifive-unreplicated-backup/`.
>
> **The one real blocker is different:** `caplifive-system-dev` pins buildroot six commits behind,
> and making it current requires WRITE ACCESS TO `caplifive-system-dev`, which this account also
> lacks. That is the single thing the migration waits on.

## OBSOLETE — 4 commits exist on one disk only

**`capstone/caplifive-system` has 4 unpushed commits and its HEAD is on no remote.**

```
aa38112  Bump sw/buildroot to the merged monitor
d1fb1e2  Bump sw/buildroot: the region-table ceiling fix
099b954  Bump sw/buildroot: complete the chain to the running monitor
6c5c740  Bump sw/buildroot: 65536-node device tree, traced monitor, and the ioctl copy_from_user fix
```

They are gitlink bumps, which makes them look trivial and makes them the opposite: they are the
**only record of which `sw/buildroot` commit corresponds to the monitor currently running on the
board**. The monitor source itself is safe — `sbi_capstone.c` is byte-identical in
`caplifive-system-dev` — but the correspondence is not recoverable from the source.

**This must be fixed before the migration, not as part of it**, because the migration touches the
same submodule and a mistake there would take these with it.

`caplifive-system`'s `capstone-bootstrap` branch **is** on the push allowlist — the allowlist
matches branch names, and `capstone-bootstrap` is listed. That is not the obstacle.

**MEASURED 2026-09-04 — this agent CANNOT replicate them. Both remotes refuse the push:**

```
git push fork   capstone-bootstrap
  remote: Permission to project-starch/caplifive-system-dev.git denied to <this account>.
  fatal: ... The requested URL returned error: 403
git push origin capstone-bootstrap
  remote: Permission to project-starch/caplifive-system.git denied to <this account>.
  fatal: ... The requested URL returned error: 403
```

So this is a **credentials/access** problem, not a policy one, and it is exactly the case
`CLAUDE.md` describes: *"a submodule with no write access is discovered only by attempting a
push."* It was discovered by attempting it. **Someone with write access to one of those two
repositories has to push this branch**, and until they do, the correspondence between the running
monitor and its buildroot commit exists on exactly one disk.

## Current state

| | `caplifive-system` (the submodule) | `caplifive-system-dev` (the clone) |
|---|---|---|
| registered in `.gitmodules` | yes, url `caplifive-system.git` | **no** — untracked in the worktree |
| branch checked out | `capstone-bootstrap` | `master` |
| unpushed | **4 commits** | 0 |
| dirty | 1 (`sw/buildroot` gitlink) | 0 |
| `sbi_capstone.c` | 1119 lines | **byte-identical** |
| size | — | 3.9 GB |

Note that `caplifive-system` already carries `caplifive-system-dev.git` as a second remote named
`fork`. The migration is in effect already half-done by hand, which is the least safe state to
leave it in: the pointer in `.gitmodules` and the remote people actually push to have diverged.

## Interim protection taken 2026-09-04 — bundles, because pushing is impossible

Since none of the 13 commits can reach a remote, they were bundled instead:

```
~/caplifive-unreplicated-backup/
  01-caplifive-opensbi.bundle   3 unpushed
  02-buildroot.bundle           6 unpushed (incl. 17e4fb609)
  03-caplifive-system.bundle    4 unpushed
```

**Verified by restoring, not by exit status** — each bundle was cloned into a fresh repository and
the named commit confirmed present (`460f6e45e`, `80e1dcfe`, `aa38112`). A `git bundle create`
that exits 0 is not evidence the commits are in it.

**This is NOT a substitute for pushing.** The bundles sit on the same disk as the repository they
protect, so they guard against a bad checkout, a deleted branch or a botched migration — not
against disk loss. They need to be copied off the machine, and the real fix is still write access.

## Found 2026-09-05 while porting the Q-03 fix — a fifth single-copy commit, and two monitor lines

`caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi` is a checkout of `caplifive-sbi`
(its remote still named `capstone-sbi.git`, pre-rename) at **`1a926b0` "Carve gp only for images
that declare a globals region" — on no remote** until the Q-03 push chain. The package checkout of
the same repo is at `04ac643` (pushed). Common base `2f772bb`; one commit each way; both about
gp/globals delivery; **zero conflict hunks** on a trial three-way merge — complementary, not
competing. Today the QEMU firmware runs one line and the domain runs the other, and the merged
line has never been built. Whether they become one line is a monitor-semantics call for the lead.

## Proposed sequence

1. **Replicate first.** Add `capstone-bootstrap` (caplifive-system) to the push allowlist and push
   it to `fork`. Nothing else in this plan happens until those 4 commits exist somewhere else.
2. ~~**Confirm the replacement claim** by diffing the trees.~~ **DONE 2026-09-04 — and it
   changes the plan. `-dev` is BEHIND, so it cannot be adopted as-is.**

   Both repositories track exactly **16 files**, and 4 of the 5 submodule pins agree
   (`hw/anvil`, `hw/qemu`, `hw/rtl`, `sw/capstone-c`). The entire difference is **`sw/buildroot`**:

   | | `sw/buildroot` pin | date |
   |---|---|---|
   | `-dev` | `f11bf691` | 2026-08-12 |
   | `caplifive-system`, recorded | `17e4fb609` | 2026-08-18 (+3) |
   | `caplifive-system`, checked out | `80e1dcfe` | 2026-08-31 (+6) |

   Verified straight-line descent: `git merge-base --is-ancestor f11bf691 HEAD` succeeds, so
   **`-dev`'s pin is an ancestor of what we are actually building.** Adopting it would REGRESS the
   monitor chain by six commits.

   **`sbi_capstone.c` being byte-identical between the trees is true and misleading** — that is
   the `package/` copy. The firmware builds from `components/opensbi`, whose pin differs:
   `460f6e45e` here against `0dbb365c5` in `-dev`'s buildroot.

   So "full replacement" is right about the repository's *role* and wrong about its *content
   currency*. The migration must carry our buildroot pin forward, not take `-dev`'s.

2b. **The unreplicated work is THREE LEVELS DEEP, not one**, and every level is on this disk only:

   | repository | branch | unpushed |
   |---|---|---|
   | `caplifive-system` | `capstone-bootstrap` | **4** |
   | `sw/buildroot` | `capstone-bootstrap-dts-65536` | **6** (incl. `17e4fb609`, the merged-monitor bump) |
   | `components/opensbi` | (pinned `460f6e45e`, on no remote) | **3** |

   `caplifive-system`'s *committed* state already points at a buildroot commit that exists
   nowhere else, so even a successful push of the outer repository alone would produce a
   dangling reference. **Push innermost-first: `caplifive-opensbi`, then buildroot, then
   `caplifive-system`.**
3. **Repoint `.gitmodules`** to `caplifive-system-dev.git`, `git submodule sync`, and bump the
   gitlink to a commit in the new repository.
4. **Delete the standalone `capstone/caplifive-system-dev/` clone** — after step 3 the submodule
   checkout *is* that tree, and keeping both invites edits landing in the copy that no longer
   feeds the build. This also reclaims the 3.9 GB.
5. **Remove the now-dead `/capstone/caplifive-system-dev/` line from `.gitignore`** (added
   2026-09-04 when the clone's role was not yet known).
6. **Announce to every lane**: a submodule URL change requires `git submodule sync --recursive`
   and a re-fetch. A lane that misses it keeps building from the archived repository and will not
   notice, because the tree still looks right.

## The risk worth stating

The firmware build path runs through this submodule. `docs/ref/` and the `board-run` skill both
name paths under `capstone/caplifive-system/`; if the submodule *path* stays the same and only the
URL changes, those stay valid — **so keep the path**. Renaming the directory to match the new
repository name would break the board documentation for no benefit.
