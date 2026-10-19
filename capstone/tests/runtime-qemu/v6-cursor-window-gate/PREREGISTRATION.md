# The cursor-window fix on the v6 runtime: counter gate, pre-registered 2026-10-20 before any run

The fix is lane `runtime/cursor-window-v6` dc1ad0a8a6fb: two parenthesisations in `runtime/virtual/heap.c`
(`bind_slots`, `backend_pointer`), so that no intermediate cursor leaves its object. Old is dev's heap.c, the v6 landing.

## Images and kit

- **Old SDK:** v6nb-kit's SDK, built from the v6 landing tree with `-fno-builtin`. Its heap.c is dev's
  (sha 4b37a6f43e9a6063).
- **New SDK:** the same SDK with `heap.c.obj` in `libapplication-runtime.a` replaced by the lane's heap.c
  (f793819c2476cbf5), compiled by the SDK's own ninja command.
- **Recipe control:** dev's heap.c compiled by that command equals the archive member in every section except
  `.symtab`/`.strtab`, which carry the source file's name. In the new object only `group_map`, `group_nested`,
  `prepare_free`, `free`/`__libc_free` and `realloc` change.
- **Programs:** `nested-churn.c` (8f75c56688d3e5d2, lane v6-cursor-window 3b1e33210dae), built with each SDK's
  `capstone-cc -O1`: nc-old d7b0ad9c524b2c15, nc-new 247413702490a65f.
- **Kit qkit7-v6wc:** the v6 launcher 2b4c20424874985b and module d073889f18f89cd1 (the v6 gate's), with the
  window-counter QEMU 4db34d9695184bcd (capstone-qemu cap-window-counter on b290f5656600). Compact profile: exact bounds
  off, `svadu=false` and `x-capstone-window-count=true` appended. Every applied -cpu string is logged.

## Static census (negcursor-census.py, raw words)

| image | instructions parsed | CINCOFFSET | negated own cursor | negated base |
|---|---:|---:|---:|---:|
| nc-old | 28,814 | 838 | 2 | 2 |
| nc-new | 28,810 | 834 | 0 | 0 |

Old's sites:
- OWN 0x1f5ac in `__capstone_malloc_group_map` and OWN 0x1f7d0 in `__capstone_malloc_group_nested`. Each is `bind_slots`
  inlined: `group - base` is formed once before the slot loop and stays live across it. The loop's SHRINK takes
  `that + start`, which is back in window.
- BASE 0x1fc34 in `__capstone_malloc_prepare_free` and BASE 0x202c4 in `free`. These are `backend_pointer`: two
  CINCOFFSETs, with the out-of-window value live for one instruction.

## The run

- One boot per run: nested-churn 3,000 rounds, `CAPSTONE_WINDOW_FILE` set per boot.
- Ten boots alternating old, new, old, ..., five each.
- Readings per boot: NC_DONE, exit, CAPSTONE_WINDOW_COUNT (escape_out, op_out), and every CAPSTONE_WINDOW_SITE line.
- **The pc mapping.** A site pc maps to image VA as `(pc & 0x7ffff) + 0x10000`, because the launcher places code on a
  512 KiB boundary (every code= base recorded so far is). A site that does not land on an instruction inside a FUNC
  symbol voids the mapping and the run.

## Predictions

1. **Controls.** Every boot: the applied -cpu string carries `x-capstone-window-count=true`, the window file is
   present with its CAPSTONE_WINDOW_COUNT line, `NC_DONE rounds=3000`, and exit 0. A boot missing any of these is
   VOID.
2. **Old.** Over the five boots, at least one escape site falls in a census window:
   - `group_map` [0x1f5ac, 0x1f628);
   - `group_nested` [0x1f7d0, 0x1f84c);
   - either backend_pointer pair (0x1fc34, 0x202c4).

   If old reads zero there, the run cannot tell old from new, and the gate is VOID: it is not a pass for the fix.
3. **New.** Over the five boots, zero escape sites and zero op sites in `__capstone_malloc_group_map`,
   `__capstone_malloc_group_nested`, `__capstone_malloc_prepare_free` and `free` (the image's FUNC symbols of the
   functions the recipe control shows changing; `realloc` and `__libc_free` are not FUNC symbols in nc-old).
4. **Op sites in those functions:** 0 on old as well. No bounds-deriving op reads the out-of-window value, since the
   loop's SHRINK sees `that + start`. One there on old is reported as a mechanism different from the census's.
5. **Sites elsewhere** (nested-churn's own code, the rest of the runtime) are reported for both images and are not
   attributed to this fix. The escape count is an upper bound, because dead temporaries are counted.
