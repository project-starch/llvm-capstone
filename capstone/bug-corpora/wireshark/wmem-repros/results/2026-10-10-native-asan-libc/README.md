# native-detect (ASan) on wmem as glib gives it its memory -- 2026-10-10

    bash capstone/bug-corpora/wireshark/wmem-repros/runners/run-asan.sh <fresh outdir>

wmem is built hosted with `WM_LIBC_SYSTEM=ON` and `WM_CHUNKS=OFF`
(`ports/wireshark/wmem/src/shared/backing.c`): `g_malloc`/`g_free` are the host's `malloc`/`free` at
the requested size. The earlier arm (`../2026-10-09-native-asan/`) ran the hosted bump arena -- one
384 MiB `aligned_alloc`, `g_free` a no-op -- where no reading but "silent" was possible.

**Result: 22 of 22 SILENT, with wmem's own positive control reporting in the same run**: control 90
(`controls/sublet-malloc/`, a jumbo `wmem_free_all` hands to `g_free`, then read) reported
`heap-use-after-free` at `wm_probe`, built with the same options. Two-sided: the same control built on
the old arena (no `WM_LIBC_SYSTEM`) printed `VERDICT DEFECT-REPRODUCED` with ASan silent -- the old arm
could not have reported even the one sequence that does reach `free()`.
