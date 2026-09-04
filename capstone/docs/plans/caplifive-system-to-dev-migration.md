# Proposal: retire `caplifive-system`, make `caplifive-system-dev` the submodule

**Status: PROPOSAL (2026-09-04), for review. Nothing is implemented.** It changes a submodule URL,
which every lane and every fresh clone is affected by, so it is not a change to make unilaterally.

Raised by the project lead: **`caplifive-system-dev` is a full replacement for
`caplifive-system`, and `caplifive-system` should be archived and no longer used.**

## URGENT AND SEPARATE FROM THE MIGRATION — 4 commits exist on one disk only

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

## Proposed sequence

1. **Replicate first.** Add `capstone-bootstrap` (caplifive-system) to the push allowlist and push
   it to `fork`. Nothing else in this plan happens until those 4 commits exist somewhere else.
2. **Confirm the replacement claim** by diffing the trees, not just `sbi_capstone.c` — the two
   share no history, so a file-level comparison across `sw/` and `hw/` is the only way to know
   what `-dev` does *not* carry. Record the diff.
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
