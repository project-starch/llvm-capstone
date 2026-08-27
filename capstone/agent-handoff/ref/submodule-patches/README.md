# Submodule changes that live here because they cannot be pushed

A patch in this directory is a submodule commit the parent repo depends on and that
we have **no write access to push** to the submodule's own remote. Attempting the
push is how you find that out, and it is the only way:

    git -C capstone/caplifive-buildroot push --dry-run origin <branch>
    remote: Permission to project-starch/caplifive-buildroot.git denied
    fatal: ... The requested URL returned error: 403

The parent's submodule pointer is therefore **deliberately NOT bumped** to those
commits. A pointer to a commit that exists on no remote makes the branch
unclonable, which is worse than a stale pointer. The patches are the portable form.

## caplifive-buildroot-0001..0004: gp-captable delivery and domain sizing

Branch `xlang-gp-captable-delivery`, four commits, 2026-08-19. Local HEAD
`34281b1`; the parent records `6912474`.

    0001  modcapstone: deliver the gp-captable init descriptor into dom_data
    0002  Scale the domain region with the image, so an interpreter has room to recurse
    0003  Raise the buddy allocator's maximum order so interpreter images fit a domain
    0004  Let a domain declare what it needs, and give it two regions instead of one

**Every domain result on this branch was produced with these applied**, SQLite and
WAMR included. Without 0004 the kernel module sizes headroom as
`max(2*code_len, 512 KiB)`, and a 265 KB interpreter image does not fit; that is
the sizing failure that once read as a compiler regression. A checkout of the
parent alone reproduces none of it.

To restore:

    cd capstone/caplifive-buildroot
    git checkout -b xlang-gp-captable-delivery 6912474
    git am ../agent-handoff/ref/submodule-patches/caplifive-buildroot-000*.patch

The `From:` headers are stripped, so `git am` attributes the commits to whoever
applies them. That is deliberate: the name rule forbids carrying an author identity
into this repo, and the first version of this commit was BLOCKED by
`precommit-scan.sh` for exactly that, having shipped four `From: <name> <email>`
lines. The patches apply unchanged either way.

If write access to that remote ever appears: push the branch and bump the parent
pointer instead, and push the SUBMODULE first, then the parent, so the parent never
references a commit that does not exist remotely.
