# CPython pymalloc defect corpus

Consumer-side temporal-safety defects in CPython whose memory comes from
pymalloc, so that no `free()` reaches `malloc` and a malloc-level tool has no
event to see. Same design as `../../postgres/mmgr-repros/`, which is the working
template: a native arm on x86 and a paired spatial/Sublet arm in a domain, one
defect per boot, the fault PC checked against a labelled probe.

The inventory, the triage and the numbers are in
`docs/ref/cpython-pymalloc-defects.md`. **23 of the defects live in the pinned
3.13.7 are reachable with the existing pymalloc port** — 21 proven by their
backports, 2 from the never-backported group after an apply-test against a
pristine `v3.13.7` tree, with 2 more unresolved.

The template's own eight arms — `shared/defects.c`, `shared/run-defects.py` and
two QEMU result sets — are on PRs #54 and #55 and are not on this branch yet;
this branch carries only the one PostgreSQL case that reached `dev`. Read the
template there, not from the tree here.

## What is different from the PostgreSQL corpus

**Three allocator layers, not two.** A freed object may stop at a per-type free
list (ten of them) and never reach pymalloc at all; the parser has its own arena
besides. The port covers pymalloc only, so every case must record which layer its
object came from — `case.json` carries it.

**A size threshold that decides whether the case is blind at all.** pymalloc
serves requests up to 512 bytes; above that `PyObject_Malloc` falls through to
`malloc` and ASan *does* report the use-after-free. Buffer-carrying defects are
therefore blind on a small input and visible on a large one. A driver must
allocate at or below the threshold, and say so, or the specimen quietly stops
demonstrating anything.

**Upstream states the problem itself**, in `Doc/using/configure.rst`: to use
AddressSanitizer you should combine it with `--without-pymalloc`, "to disable the
specialized small-object allocator whose allocations are not tracked by ASan".
That is the corpus's thesis in the project's own build documentation.

## Layout

    <gh-NNNNN>_<slug>/
        case.json        machine-readable claims: layer, size, arms, oracles
        before.c         native arm, two modes: --detect and --damage
        after.c          domain arm, via the port's corpus seam
        PROVENANCE.md    upstream commit, quoted hunks, what is real and what is reduced
        README.md

    shared/              the domain program and the paired runner
    results/<stamp>/     matrix.tsv and input hashes; never raw serial captures

## Status

Scaffolding and the first case's provenance only. No driver is written yet, and
nothing here has been run. The PostgreSQL corpus is the template to follow, down
to the control discipline: a native arm claims nothing from ASan's silence, and
the runner exits 75 with no verdict if its positive control fails to fire.
