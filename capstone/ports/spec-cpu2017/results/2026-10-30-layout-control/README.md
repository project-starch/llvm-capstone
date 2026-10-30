# Layout control across the arms, 2026-10-30

`tools/layout-check.py` output for the heap-last link (`CAPSTONE_SPEC_LAYOUT=heap-last`, README "Code layout across
the arms") and for its controls. Toolchain, SDKs and SPEC tree as the campaign's: clang b90e13006a01, the arms'
SDKs perf/sdk-T1 (capstone-spatial) and perf/sdk-T3 (capstone-lifetimes), kit at spec-silicon cd44551c2ae4 plus this
change.

| File | Build | Reading |
|---|---|---|
| `heap-last-T1-T3.txt` | heap-last, every manifest benchmark, both arms (build.sh's own run of the check) | 13 of 13 pairs PASS |
| `heap-last-heapfix-T1-T3.txt` | heap-last, 505.mcf_r and 557.xz_r, the heapfix SDK pair (perf/sdk-heapfix-T1, -T3) | 2 of 2 PASS |
| `control-uncontrolled-T1-T3.txt` | the campaign build of 2026-10-28 (the SDK's own link.ld) | 0 of 13 PASS; mcf main 0x212ec vs 0x21414 |
| `control-planted-shift4.txt` | heap-last mcf, capstone-lifetimes relinked with 4 bytes after .text.entry (`layout-sdk.py --plant-shift 4`), against capstone-spatial | FAIL, 38 of 38 benchmark functions moved by 4 |
| `control-symbol-inside-region.txt` | heap-last mcf, both arms with a function symbol added inside the heap text region (`llvm-objcopy --add-symbol`) | FAIL, heap_regions 2 (one per image); no address mismatch |
| `control-region-marker-removed.txt` | heap-last mcf lifetimes against itself, `__layout_heap_text_start` renamed | FAIL, heap_regions 28 (heap.c's code outside every region) |

The planted relink used the build's own link line; without the plant it reproduces the build's image byte for byte
(sha256 d5bb8645d28c887c...). With `CAPSTONE_SPEC_LAYOUT` unset, 505.mcf_r rebuilt in a fresh tree reproduces the
campaign images of both arms byte for byte (8fbf5fe40f1fe8e2... and 9b7507f16a950592...).

Image hashes of the heap-last build (both arms) are in its build record; mcf: capstone-spatial 53342c7e153a5eb1...,
capstone-lifetimes d5bb8645d28c887c....

QEMU twin, `twin-505.mcf_r-test.txt`: the heap-last mcf images (the hashes above), staged with `board/make-stage.py
--strip` from a `--fake` test run that named each arm's layout SDK, run in one boot on kit-q34 (`board/run-stage.py
run --backend qemu`, `--twin` the census-q34 run of the campaign images). Both arms exit 0, specdiff 2/2 PASS each,
outputs identical across the arms and identical to the campaign images' twin outputs. instret 35,301,323,506
(capstone-spatial) and 35,302,833,757 (capstone-lifetimes); the campaign images read 35,301,316,585 and 35,302,835,720
on the same kit, so the layout moved each by under 7,000 instructions (0.00002 %).
