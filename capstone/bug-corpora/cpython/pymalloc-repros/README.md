# CPython pymalloc defect corpus

Consumer-side temporal-safety defects in CPython whose memory comes from
pymalloc, so that no `free()` reaches `malloc` and a malloc-level tool has no
event to see. Same design as `../../postgres/mmgr-repros/`: one defect per boot,
a paired spatial/Sublet arm in a domain, the fault PC checked against a labelled
probe.

**Eight of them run, 16/16 arms passing, with a negative control that fires.**
See `results/20260918-qemu/`.

The inventory, the triage and the numbers are in
`docs/ref/cpython-pymalloc-defects.md`. **23 of the defects live in the pinned
3.13.7 are reachable with the existing pymalloc port** — 21 proven by their
backports, 2 from the never-backported group, with 2 more unresolved. Eight of
the 23 have drivers here; the rest do not yet.

## The eight

| # | upstream fix | consumer | allocator shape |
|---|---|---|---|
| 0 | `gh-143543` | `itertools.groupby` | free / reuse / stale read |
| 1 | `gh-146613` | `itertools._grouper` | free / reuse / stale read |
| 2 | `gh-142829` | `Context.__eq__`, `Python/hamt.c` | interior pointer into a freed block |
| 3 | `gh-142831` | the JSON encoder's items list | stale entry reached through a live array |
| 4 | `gh-145244` | the JSON encoder's dict key | bulk free, stale read on the error path |
| 5 | `gh-148660` | `OrderedDict.copy()` | **pointer load out of a freed block, then followed** |
| 6 | `gh-151295` | `bytes.join()` via `__buffer__` | payload buffer, pinned sub-512 |
| 7 | `gh-148395` | `{LZMA,BZ2,_Zlib}Decompressor` | cursor surviving across two API calls |

Seven are proven live at the pin by their own backports into 3.13 after our tag.
Case 4 is not: it was never back-ported, and the apply test fails only because
upstream renamed the function. It is live by inspection — `Modules/_json.c:1621`
of `v3.13.7` hands the borrowed key straight on with no `Py_INCREF` at all. Its
`PROVENANCE.md` quotes the pinned source.

**Cases 0, 1, 3 and 4 reduce to one allocator sequence.** They are four
separately reported upstream defects in three modules, and that sameness is the
point rather than a redundancy: one revocation mechanism covers a class upstream
has been fixing one module at a time. Case 5 is the most interesting of the
eight — the loop reads `node->next` *out of* the freed node, so the unprotected
outcome is not a crash but a plausible wrong answer.

## What is different from the PostgreSQL corpus

**Three allocator layers, not two.** A freed object may stop at a per-type free
list (ten of them) and never reach pymalloc at all; the parser has its own arena
besides. The port covers pymalloc only, so every case records which layer its
object came from — `case.json` carries it.

**A size threshold that decides whether the case is blind at all.** pymalloc
serves requests up to 512 bytes; above that `PyObject_Malloc` falls through to
`malloc` and ASan *does* report the use-after-free. This binds case 6, which
carries a payload buffer: pinned small it is invisible, grown large it is an
ordinary heap use-after-free. The other seven hold pointers to object structs and
are below the threshold on any input.

**Upstream states the problem itself**, in `Doc/using/configure.rst`: to use
AddressSanitizer you should combine it with `--without-pymalloc`, "to disable the
specialized small-object allocator whose allocations are not tracked by ASan".
That is the corpus's thesis in the project's own build documentation.

## Layout

    <gh-NNNNN>_<slug>/
        case.json        machine-readable claims: layer, size, shape, arms, oracles
        PROVENANCE.md    the upstream hunk, quoted; what is real and what is reduced
    shared/defects.c     the domain program, one case per boot
    shared/run-defects.py the paired runner and its oracles
    tools/               the survey scripts behind the inventory's numbers
    results/<stamp>/     matrix.tsv and input hashes; never raw serial captures

## Running it

    cmake -S <port> -B <build> ... -DPY_CORPUS_SRC=<abs path>/shared/defects.c
    cmake --build <build> --target defects
    python3 shared/run-defects.py <out> --domain-build <build> --linux-build <guest>

and the control that makes the result mean something:

    python3 shared/run-defects.py <out> --cases 0 --negative-control

## What is NOT here

- **No native arm.** The PostgreSQL corpus pairs its domain arms with an x86
  `before.c`; this one does not yet. A native arm must claim nothing from ASan's
  silence — that claim is tautological, because the memory never reached
  `malloc` — so it needs a Valgrind run to say anything, and Valgrind is not
  installed on this host.
- **No containment result.** Every Sublet row reads `delivered: false`: this QEMU
  halts the domain rather than delivering the fault. The fault is raised; the VM
  surviving it is not shown here.
- **Fifteen of the 23 reachable defects have no driver.**
