# The twenty defects in a domain: the plain interpreter against pymalloc under Sublet

Measured 2026-09-25 on QEMU. Each case ran once in each arm, with its own boot.

## The two arms

Both images come from one tree and one compiler (hashes in `inputs.json`):

- **control**, the plain port as merged: patches 0001-0013, with pymalloc as upstream has it plus
  provenance.
- **sublet**: 0001-0008 and 0010-0013, with 0014 in place of 0009. Every pymalloc block is issued
  and revoked by the adapter (`ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c`), and
  the domain runs with `CPY_SUBLET_MODE=1`.

The compiler is `compiler/c66-machinecse-pre-trapping-cap-arith`. Without C-66 the control arm of
six of these cases ended at a compiler-made trap in `_PyArg_UnpackKeywords` before the defect ran
(`docs/ref/ISSUES.md`, C-66).

The scripts are `../../cases/caseNN.py`. Each is wrapped in markers:

- `CPY-CASE-BEGIN`, then the imports the harness needs;
- `CPY-CASE-ARMED`, then the case itself;
- `CPY-CASE-END`.

Case 11 also carries the America/Los_Angeles zone file, because the domain stdlib has no tzdata.
That is harness only, not the defect.

## How a row is read

- **A catch** is a halt after `CPY-CASE-ARMED` and before `CPY-CASE-END`, and not in the
  interpreter's shutdown walk.
- **A control halt counts as a capability catch only with a Capstone diagnostic** (`Cap mem
  access ...`). Cause 2 is an illegal instruction: a jump into data that the machine did not name.
- **Not reachable** means the control prints the case's own `trigger-NN ...: <why>` exit, or an
  import fails.
- **Placing a fault.** The ELF address is `pc - pcc base + 0x10000`, symbolised against the image
  that ran. `operand` is what the faulting register held:
  - `tag-cleared capability`: the upper half is still a capability's metadata and the tag is gone.
    That is what a revoked or overwritten capability looks like.
  - `integer`: the upper half is 0.

Serial captures are not committed, because a capture carries account names. The per-case lines
here come from them through `emit-results.py`, whose rules are the ones above.

## Result

| verdict | cases | n |
|---|---|---|
| CAUGHT by Sublet only | 0 2 5 6 9 10 12 14 15 16 19 | 11 |
| caught without Sublet too | 1 3 8 11 | 4 |
| not reachable in the domain | 7 13 17 18 | 4 |
| not caught | 4 | 1 |

Every case, from `matrix.tsv`:

| case | fix | control (plain) | sublet | verdict |
|---|---|---|---|---|
| 0 | gh-143543 | completes | cause 24 _PyEval_Vector+0x64 [tag-cleared capability] | CAUGHT by Sublet only |
| 1 | gh-146613 | cause 24 find_name_in_mro+0xdc [integer 0x1] | cause 24 PyObject_RichCompare+0x338 [tag-cleared capability] | caught without Sublet too (capability fault in the control) |
| 2 | gh-142829 | completes | cause 24 hamt_iterator_next+0x88 [tag-cleared capability] | CAUGHT by Sublet only |
| 3 | gh-142831 | cause 24 _PyObject_Malloc+0x94 [tag-cleared capability] | cause 24 encoder_listencode_obj+0xb0 [tag-cleared capability] | caught without Sublet too (capability fault in the control) |
| 4 | gh-145244 | completes | completes | not caught |
| 5 | gh-148660 | SystemError: null argument to internal routine | cause 24 odict_copy+0x2e4 [tag-cleared capability] | CAUGHT by Sublet only |
| 6 | gh-151295 | completes | cause 24 slot_bf_getbuffer+0x21c [tag-cleared capability] | CAUGHT by Sublet only |
| 7 | gh-148395 | ModuleNotFoundError: No module named 'zlib' | ModuleNotFoundError: No module named 'zlib' | not reachable in the domain (no module zlib) |
| 8 | gh-112127 | cause 24 atexit_delete_cb+0x30 [integer 0x0] | cause 24 atexit_delete_cb+0x30 [integer 0x0] | caught without Sublet too (capability fault in the control) |
| 9 | gh-139210 | completes | cause 24 strlen+0x0 [tag-cleared capability] | CAUGHT by Sublet only |
| 10 | gh-142560 | ValueError: empty separator | cause 24 memchr+0x1c [tag-cleared capability] | CAUGHT by Sublet only |
| 11 | gh-142783 | cause 5 unicodekeys_lookup_unicode+0xd4 [out of bounds] | cause 24 _Py_dict_lookup+0x98 [tag-cleared capability] | caught without Sublet too (capability fault in the control) |
| 12 | gh-143004 | completes | cause 24 slot_nb_add+0x3ac [untagged (cincoffset)] | CAUGHT by Sublet only |
| 13 | gh-144833 | ModuleNotFoundError: No module named '_ssl' | ModuleNotFoundError: No module named '_ssl' | not reachable in the domain (no module _ssl) |
| 14 | gh-146011 | completes | cause 24 signaldict_repr+0xd0 [tag-cleared capability] | CAUGHT by Sublet only |
| 15 | gh-149449 | cause 2 PyUnicodeDecodeError_GetReason+0x64 [wild jump (illegal instruction)] | cause 24 PyCodec_NameReplaceErrors+0x260 [tag-cleared capability] | CAUGHT by Sublet only |
| 16 | gh-151403 | completes | cause 24 PyOS_FSPath+0x2d0 [tag-cleared capability] | CAUGHT by Sublet only |
| 17 | gh-151416 | exits | exits | not reachable in the domain (os.spawnv is the pure-Python fallback on this platform (linux); the C) |
| 18 | gh-151695 | exits | exits | not reachable in the domain (curses unavailable) |
| 19 | gh-153539 | completes | cause 24 _io_TextIOWrapper_tell+0x5dc [untagged (cincoffset)] | CAUGHT by Sublet only |

**What the rows say.**

- **Sublet catches all 14 use-after-frees that ASan reports on a native 3.13.7** (cases 0, 1, 2, 3,
  5, 6, 9, 10, 11, 12, 14, 15, 16, 19). Each catch comes after `CPY-CASE-ARMED`, in the defect's own
  function, for example `hamt_iterator_next`, `odict_copy`, `slot_bf_getbuffer`, `signaldict_repr`,
  `PyOS_FSPath` and `_io_TextIOWrapper_tell`. The faulting operand is a capability whose tag was
  cleared, or an untagged `cincoffset` into the Sublet arena.
- **Eleven of those the control does not stop at all.**
  - Eight run to their end marker.
  - Cases 5 and 10 end in a wrong Python exception: the corruption was consumed silently.
  - Case 15 dies on a jump into data (cause 2), which is no diagnosis.
- **In 1, 3 and 11 the capability machine faults without Sublet as well, but later or elsewhere:**
  - 1 loads through the integer 1;
  - 3 faults in `_PyObject_Malloc` on the corrupted free list;
  - 11 is a bounds fault.
  Sublet stops each of the three at the use.
- **Case 8** is a NULL dereference, the same in both arms, and ASan's SEGV at 0. Sublet adds nothing.
- **Case 4 is not a miss.** The pinned 3.13.7 has the unsafe pattern (a key and value borrowed
  across `default()`) but not the read after the free. That read is `_PyErr_FormatNote("%R", key)`,
  which exists only on `main`. The dangling pointers are revoked and never used, which is correct.
  `case.json` calls the case live by inspection; the pattern is live, the use is not.
- **Case 16 is reachable**, although the native matrix marks it otherwise: the use-after-free is in
  the argv conversion, in the parent, before any fork.
- **Cases 7, 13, 17 and 18 are not reachable here:**
  - 7 needs `zlib`;
  - 13 needs `_ssl`;
  - 17 is `os.spawnv`'s pure-Python fallback on Linux;
  - 18 needs `curses`.

## The negative control, and a port defect it found

`run-control-benign.py` drives the same modules the ordinary way, with no defect. Its purpose is to
catch a fault that belongs to the Sublet arm itself and not to a use-after-free.

- itertools, contextvars/hamt, json, OrderedDict and `bytes.join` ran without a fault in both arms.
  With a collection after each section, the GC traversed them all as well.
- Both arms then fault identically, on `_Py_NoneStruct` held **untagged** (`value_hi = 0`):
  `visit_decref+0x0` during a collection, or `element_gc_clear+0x8c` with the collector off.
- **Cause: `Modules/_elementtree.c:57-59`.** `JOIN_SET` and `JOIN_OBJ` keep a flag bit in
  `text`/`tail` through `uintptr_t`, and no port patch touches the file. With the flag set the tag
  is lost, and `text`/`tail` is usually `None`.
- The section markers lag the fault by one section, because the last stdout hostcall round is not
  printed when the domain halts. That is why a bisect first pointed at `atexit`; `atexit` alone
  runs clean.

This is a defect of the interpreter port, in both arms, and not a Sublet false positive. It is
open, and gets its own patch. The control's modules after ElementTree (bytearray, zoneinfo,
Counter, decimal, unicodedata, fspath, TextIOWrapper, fastsearch) are therefore **not yet covered**.

None of the twenty cases shows this signature, which is an untagged `None` in `visit_decref` or an
element clear. Every catch above is in the defect's own function, on a tag-cleared capability or an
arena address.

## Limits

- **One run per arm**, with retries on stalls. The interpreter is deterministic here, but a guest stall is not, and stalls
  were frequent: at the module swap, after the domain loads, and mid-run. A stalled attempt is
  retried, up to three times, and never read as a result.
- **"Completes" in the control means the script reached its end marker.** Whether the stale access
  also happened there is not observed; the script and the native ASan arm
  (`../20260925-native-asan/`) say it did.
- **Which allocator layer a freed object came from is still a proxy** (`case.json`). A catch shows
  that the block reached pymalloc; a miss does not show that it did not.
- QEMU runs with `CAPSTONE_REV_NODES=8388608`. The default node pool, 65536, is exhausted by the
  Sublet arm.
