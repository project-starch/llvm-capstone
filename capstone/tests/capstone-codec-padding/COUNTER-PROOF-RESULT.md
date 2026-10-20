# -capstone-pad-globals: the counter proof's result (2026-10-20), against COUNTER-PROOF-PREDICTION.md

Run: memcached plain-temporal case 00 + controls, run-virtual-cases.py (wt-gate), kit qkit7-wc9-nr (capstone-qemu
cap-window-counter 8c61567c9b, binary 3c811281863a601a: 16384 sites per kind; module a1fa252f; norandmaps).
- OFF: SDK t3-freeze-nb-nomovc (clang a5be1a4d); case image b07337346ca79c07.
- ON: the same recipe on clang llvm-padg (this branch) with CAPSTONE_VIRTUAL_PAD_GLOBALS=ON.

| counter (context unless noted) | OFF | ON |
|---|---|---|
| result_checked | 3345 | 3345 |
| result_inexact (sites; sum of n; overflow) | 398 (16; 398; none) | 0 (0; 0; none) |
| escape_inexact | 13 | 0 |
| result_m (the kit's M-mode monitor) checked / inexact | 106 / 35 | 106 / 35 |
| op_inexact (the monitor's, at boot) | 32 | 32 |
| case 00 verdict | CAUGHT, probe, cause 25, mct_read_probe+56 | identical |

Sites, mapped with code= 0x3ff7f00000 from the run's own fault line and accepted only where the instruction at the
mapped VA is a SHRINK (the case image and controls.dom map identically: the runtime links at the same offsets):
15 of 16 accepted, all forming large globals - __capstone_malloc_find 240, _slot_key 64, _slot_take 32, malloc 14,
_nested_release 12, _slot_release 4, _prepare_free 4, cap_malloc_validate 4 (index_heads, 64 KiB); 
__capstone_application_prepare 12, __capstone_domain_environ 4, capstone_main 4 (arguments/environment, 0x4010);
__capstone_cap_init 2. The 16th (n=2) lands on no SHRINK in either image; its first value [+0x2008) is libc's stdio
buffer, one of the inexact objects the static census lists. UNRESOLVED by pc, global by value.

Against the prediction: (1) holds - the index_heads sites are there OFF, each n > 0. (2) holds - ON reads 0
result-inexact, so 0 from global sources; the static census of the ON image has all 7 objects >= 4 KiB exact,
index_heads at a multiple of 128. (3) holds - the verdict is identical. (4) is WRONG for escape-inexact: predicted
unchanged, it fell 13 -> 0, because the context's escapes were saving those globals' inexact capabilities. The
monitor's op-inexact (32) is unchanged, as predicted.
