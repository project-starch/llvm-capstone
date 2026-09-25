# The same twenty defects, driven from Python

The corpus beside this directory is **model-consumer / real-allocator**: each
`case.c` makes the allocator calls the upstream defect makes, in the same order,
because reaching them in place needs a running interpreter and the pymalloc
component port does not put one in a domain. `cpython/8-reintegration` does —
the interpreter runs scripts in a capability domain — so the consumer half
becomes reachable for the first time, and this directory is it: **one Python
script per defect, no C model.**

    cases/caseNN.py                    one script per case, numbered as the corpus is
    results/20260925-native-asan/      what each script does on a native 3.13.7
    results/20260925-qemu-interpreter/ all twenty in a domain: the plain interpreter
                                       against the one with pymalloc under Sublet

The domain arm is `docs/plans/cpython-interpreter-sublet.md`, and its result is
`results/20260925-qemu-interpreter/README.md`.

**Provenance.** The twenty scripts were written on 2026-09-24 outside the tree
and measured here on 2026-09-25; the case numbering was checked against every
`case.json` (20 of 20 agree on the `gh-` number). Each script carries its own
docstring saying which upstream test it was reduced from.

## What they do on a native 3.13.7 — measured 2026-09-25

The interpreter is a `--with-address-sanitizer --without-pymalloc --with-pydebug`
build of the pinned 3.13.7. `--without-pymalloc` is what makes the arm say
anything at all: with pymalloc in place the memory never reaches `malloc` and
ASan has no event to see, which is the corpus's whole thesis (`../README.md`).

| outcome | cases | n |
|---|---|---|
| `heap-use-after-free` | 0 1 2 3 5 6 9 10 11 12 14 15 16 19 | **14** |
| SEGV on address 0 | 8 | 1 |
| no trigger | 4 7 13 17 18 | 5 |

The five that do not trigger say why themselves, and four of the five are not
fixable by writing a better script:

- **7** needs `_testcapi`; the decompressor's buggy error path is only reachable
  through it, and this build has `--disable-test-modules`.
- **13** needs `SSL_new()` to fail on an OpenSSL allocation failure. There is no
  Python-level trigger for that.
- **17** `os.spawnv` takes the pure-Python fallback on Linux, so the C conversion
  in `posixmodule.c` is never entered.
- **18** `curses` needs a terminal.
- **4** runs to completion. It is the one case that was never back-ported, kept
  as live-by-inspection (`04_*/PROVENANCE.md`), and the Python trigger does not
  reach it either.

Case **8** crashes but ASan calls it a SEGV at address 0 rather than a
use-after-free, so it is counted apart rather than folded into the 14.

## Thirteen of the twenty are candidates for a domain arm

`results/20260925-native-asan/matrix.tsv` carries a `domain_candidate` column.
It is "yes" when the script reproduces natively **and** our domain interpreter
can run it at all. Two things take cases out:

- **Modules the domain build does not have**, from the port's own compile survey
  (`ports/cpython/interpreter/results/survey-2026-09-24-*.txt`, "missing 13"):
  `_bz2`/`_lzma`/`zlib` (case 7), `_ssl` (13), `_curses` (18).
- **No processes in a domain.** Cases 16 and 17 end in `fork_exec` and `spawnv`.
  *Measured otherwise for 16 (2026-09-25):* its use-after-free is in the argv
  conversion, in the parent before any fork, and the domain arm reaches it. So
  the domain run takes all twenty and lets each say whether it is reachable.

That leaves **13**: itertools twice, `hamt.c`, `_json`, `odictobject.c`,
`stringlib/join.h`, `_elementtree`, `bytearrayobject.c`, `_zoneinfo`,
`_collectionsmodule.c`, `_decimal`, `unicodedata`, `_io/textio.c` — all of them
modules the survey reports compiled, and none needing a process.

## What a domain arm would settle that the C corpus cannot

The C cases call `pym_malloc`/`pym_free` directly, which **assumes** the freed
object reaches pymalloc. Through the interpreter that becomes a measurement:
CPython stacks ten per-type free lists above pymalloc, and
`docs/ref/cpython-pymalloc-defects.md` says the layer column is a proxy taken
from the file each fix touches, names its own count a lower bound, and flags
`Context.__eq__` (case 2) as likely free-list. Every case's `case.json` says
`allocator_layer: pymalloc`; a domain arm is where that is either confirmed or
refuted per case.
