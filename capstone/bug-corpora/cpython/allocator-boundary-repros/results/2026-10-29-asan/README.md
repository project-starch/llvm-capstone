# 2026-10-29: pymalloc-repros' remaining nine issues, run whole on the host

`../../../pymalloc-repros` holds 20 temporal defects as C models against the real `obmalloc.c`. Eleven of
its issues already had an interpreter form here (cases 22-32). This run asked of the other nine: is
the defect reached when the whole interpreter runs upstream's own reproducer, and can that run on the
Capstone build? Three can, and became cases 33-35. Six cannot, for the reasons in the last section.

## Builds

Both from `Python-3.13.7.tgz` (sha256 `6c9d8083...53b2`, the port's `upstream.json`), on p13 with gcc
15.2.0, `ASAN_OPTIONS=detect_leaks=0` for the build and every run (the frozen-module step aborts on
LeakSanitizer otherwise):

| build | configure | pymalloc | `python` sha256 |
|---|---|---|---|
| liveness and sites | `--with-address-sanitizer --without-ensurepip` | off (configure forces it off) | `34d7a01bc0108195a01c6d1cbf546566bca9204965c1a3d73663efee17ef5c55` |
| layer | `CFLAGS=-fsanitize=address -fno-omit-frame-pointer LDFLAGS=-fsanitize=address --without-ensurepip` | on | `704c9ae6fe75d69ddb7da178bb5cbf737a312b1a0cb3f77abe74c96a5783f32b` |

`--with-address-sanitizer` sets `with_pymalloc="no"` in the configure body (configure.ac at v3.13.7),
after option parsing, so `--with-pymalloc` cannot turn it back on and `PYTHONMALLOC=pymalloc` is refused
at startup ("unknown allocator"). The layer build passes the sanitizer flags directly instead.

## Results

Each trigger ran from its case directory. Case 28 ran in the same batch as the positive control.

| case | issue | `PYTHONMALLOC=malloc` | `PYTHONMALLOC=pymalloc` (layer build) | negative control |
|---|---|---|---|---|
| 28 (control) | gh-143543 | heap-use-after-free, 328-byte region | no heap report; SEGV at 0x8 on the free-list link | clean |
| 33 | gh-151295 | heap-use-after-free, 328-byte region | no heap report; SEGV at 0x8 on the free-list link | clean |
| 34 | gh-151403 | heap-use-after-free, 328-byte region | no heap report; TypeError, completes | clean |
| 35 | gh-153539 | heap-use-after-free, 41-byte region | no heap report; completes | clean |

The malloc column is each case's liveness proof, and the layer column puts all three on the nested side
(README.md, "The side is measured"). `sites.tsv` is `probe/asan-sites.py run` on the first build with
`--only 33,34,35,28`, case 28's row left out because its sites are recorded from 2026-10-10. It gave
exactly those sites again, `Py_INCREF,_PyEval_Vector`.

Case 34 is the issue's reproducer with one argument appended. The issue was written against main,
whose `fork_exec` takes 22 arguments, while 3.13.7 takes 23, the last being `allow_vfork`. The first run
without it raised the argument-count TypeError before converting anything, and printed no report. It is
not counted.

## The five that do not run whole

| pymalloc-repros case | issue | why not |
|---|---|---|
| 13 | gh-144833 | Needs `_ssl`, which the Capstone build lacks (no OpenSSL). The freed object is read only when `SSL_new()` returns NULL, and the issue forces that with a preloaded hook. |
| 18 | gh-151695 | Needs `_curses` (no ncurses for the target) and a terminal for `initscr()`. |
| 17 | gh-151416 | Not reachable on Linux. `os_spawnv_impl` is compiled only under `HAVE_SPAWNV`, which `Modules/posixmodule.c` defines only under `_MSC_VER`. On Linux `os.spawnv` is the Python fallback in `Lib/os.py`, and the fix's own test is skipped there (`requires_native_spawnv`). |
| 04 | gh-145244 | Upstream's regression test is silent at the pin. The stale read the fix describes is `_PyErr_FormatNote("%R", key)`, which `encoder_encode_key_value` does not have at v3.13.7. The fix landed on main only. The test body (scratch copy of this case's would-be `trigger.py`) raised its expected TypeError on the liveness build with no report. |
| 08 | gh-112127 | Upstream's regression test is silent at the pin. pymalloc-repros quotes main's `atexit` code, a Python list of callback tuples. 3.13.7 keeps an array of `atexit_py_callback` structs. The test body ran both of its actions on the liveness build with no report. |

Case 07 (gh-148395, the zlib decompressor) was in this table on 2026-10-29 morning and came off it the
same day: it runs whole as case 36. It needed zlib cross-built into the CPython port (the port had no
zlib) and the MemoryError injector its defect lives on, CPython's own `set_nomemory`, supplied as the
`_capstone_nomemory` module because the build omits the 32-file `_testcapi`. Its liveness has a
fixed/buggy differential: on the pinned build the trigger reports `memcpy-param-overlap` at
`decompress` (zlibmodule.c:1585); with the upstream fix applied it runs clean.

The last three rows bear on pymalloc-repros' own claim that all 20 of its defects are live at the pin.
Its case 17 rests on a back-port to the 3.13 branch, and the back-ported code is Windows-only. Its cases
04 and 08 model a stale access that upstream's own test does not produce at 3.13.7. That corpus is not
changed by this run. Whether its three cases leave the count is the lead's decision.
