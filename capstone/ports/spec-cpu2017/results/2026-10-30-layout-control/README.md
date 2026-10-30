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

The planted relink used the build's own link line; without the plant it reproduces the build's image byte for byte
(sha256 d5bb8645d28c887c...). With `CAPSTONE_SPEC_LAYOUT` unset, 505.mcf_r rebuilt in a fresh tree reproduces the
campaign images of both arms byte for byte (8fbf5fe40f1fe8e2... and 9b7507f16a950592...).

Image hashes of the heap-last build (both arms) are in its build record; mcf: capstone-spatial 53342c7e153a5eb1...,
capstone-lifetimes d5bb8645d28c887c....
