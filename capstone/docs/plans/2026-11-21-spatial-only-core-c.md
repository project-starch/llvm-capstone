# Spatial-only core, step C: pre-registration of the full build (2026-11-21)

Written before the synthesis-only gate and the full build. capstone-ariane `spatial-only-c` 321af31a3 = step B
1f1ed11f9 + one decoder change (ace0ae890): REVOKE and MREV are LEGAL again and go to the stub ("complete, no
effect"), as in step A (cf7a00ce6). DROP stays illegal. Plus the monitor-shape arm sp-monshape.S (tests only).

## RETRACTION, carried from step B

Step B's pre-registration (llvm-capstone land/rtl-spatial-prereg 9dacb7d2658e) said: "Only REVOKE, MREV and DROP are
zero in every binary the +Spatial board stage runs". **False.**
- The census behind it listed the monitor's operations by name, and capstone-c's `__mrev()` / `__revoke()` builtins
  emit raw encodings that such a census cannot see.
- Step B's first board boot (board 1, 2026-11-21) wedged in the monitor's `create_domain`, right after its DBAS/DENT
  traces (capstone-sbi 37ebb55 `sbi_capstone.c:2178-2201`), at the `__revoke` (cached domain) or `__mrev` (managed
  domain) that step B had made illegal.
- In simulation, sp-monshape on 1f1ed11f9 reproduces it without the monitor: the managed branch is a silent M-mode
  trap loop, and the cached branch takes cause 2 at the MREV.
- Step B's seal and its keys stand; the defect was the illegal set, not the build.

## The corrected census, by ENCODING

Decoder: `capstone/tools/lineb-monitor/lifetime-op-scan.py` (opcode 0x5b, funct3 1, funct7 and the fixed operand
fields, every 2-byte alignment). It is one decoder for every file. Positive control: sp-illegal.o reads 1 REVOKE,
1 MREV and 1 DROP. Files: every file in the spatial board run's bake manifest
(out-nat-spo-frcdkmsspa3-b1/502.gcc_r-train01-native), the whole fw_payload.bin, the kernel Image and the stock
kernel and module.

| binary | REVOKE | MREV | DROP | where (symbolised for the monitor) |
|---|---|---|---|---|
| monitor (fw_payload.elf, capstone-sbi 37ebb55 + CSC_SPATIAL_ONLY) | 6 | 8 | 0 | MREV: _create_domain, _loan_begin, _shared_region_annotated x2, _share_child_region, _managed_destroy_domain; REVOKE: _create_domain, _loan_end, _revoke_region x2, _managed_destroy_domain, _managed_reset_region |
| task-driver | 0 | 2 | 0 | |
| gcc-coverage.dom, io-direct.dom, io-exchange.dom, probe.dom | 1 each | 0 | 0 | |
| kernel Image | 0 | 1 | 0 | at 0xb3af44; the stock kernel has the same word (0xb3acc4), so most likely data |
| blocked.dom, compute.dom, cpugcc_r_base.native, capstone.ko, the rest | 0 | 0 | 0 | |
| fw_payload.bin, whole (monitor, kernel, initramfs) | 6 | 9 | 0 | two gzip regions inside it were not decompressed or scanned |

So DROP is the only one of the three with no encoding in any image of the boot, and it is the only one step C keeps
illegal. REVOKE and MREV go to the stub.

## Pre-synthesis gates

- **sp-monshape** (pre-registered in the file before its run; readings in the file, 321af31a3). Both of the
  monitor's create branches (managed: mrev, call, revoke and reinit; cached: mrev, revoke, reinit, then the call on
  the reclaimed block) meet every criterion on the fix and on P1's tree: the domain returns, the first trap is S's
  ecall, validity is 1 and the readback is 0x5a5a.
  - The one difference: REVOKE returns LINEAR on the stub (UNINIT on P1), so a reclaimed block is not scrubbed on
    the spatial core.
  - Step B fails as above.
- **Lint and the 95-sweep:** running; their readings are added here before the gate request.

## Keys

The reference is step B's routed seal (board/meas-pw 65cdf137af75: LUT 179,661, FF 72,847, placer congestion 4, 0
loops, WNS +0.170). Step C removes two of step B's three decode terms, so the prediction is step B within noise.

| key | prediction | if missed |
|---|---|---|
| synthesis-only gate: mapping, loops | mapping peak within 10 % of step A's gate (4,914 MB), inside the 40 GB guard; 0 loops | guard kill: partition read |
| synthesis-only gate: core LUT / FF | step A's gate +- 2,500 / +- 2,000 (core -7,774 LUT / -3,696 FF against P1's gate) | per-module census |
| placed LUT / FF | 179,661 +- 2,500 / 72,847 +- 2,000 | census |
| routed | 0 errors (ROUTING-FAILURE BRANCH: placed area and congestion only, no WNS, not flashable) | |
| congestion, loops | <= 5; 0 | |
| WNS | step B's +0.170 within the 3.74 ns null band, [-3.57, +3.91] | outside: worst-path families before any flash |

## Silicon key

The monitor's first domain create and enter returns, with the same firmware variant as step B's boot (37ebb55,
CSC_SPATIAL_ONLY). Then the +Spatial rungs under the runtime lane's pre-registration.
