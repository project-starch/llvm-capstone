# CAPSTONE_HEAP_MAPPING_REUSE_HWM on the native twin: pre-registration (written 2026-11-22, before any run)

## Why

On 505.mcf_r train the paper's Full image (`fa650b98`, SDK FRCDKMSa3) is OOM-killed on the twin at 100 s. The same
image with the mapping cache compiled out (R0 `24a4cc5e`) completes, at VmHWM 885,680 kB in a 921,488 kB guest. These
are the nested lane's runs, pre-registered at 8e01dc646c0c with results at 0c97515b979d, and rechecked here against the
run directories. Full's anon-rss at the kill was 892,748 kB, 7,068 kB above R0's peak.

The change (1e5d9c1df593, `malloc-mapping.c`) makes the cache give mappings back before a fresh mapping, until
live + cached + new fits under max(peak live, live + new). The footprint then never exceeds the peak of live mappings,
which a build without the cache also reaches. With the switch off, the object is byte-identical to the measured SDK's.

## Images

Each image is FRCDKMSa3's own objects relinked by their recorded `link.txt` through a copy of its SDK, with
`malloc-mapping.c.obj` replaced in the runtime archive. The scripts are `tools/mkimg2.sh` and `tools/cc.sh`; the flags
come from the sdkm13b `build.ninja`. Controls, all byte-identical:
- unmodified source → the SDK's object;
- `-DCAPSTONE_HEAP_MAPPING_REUSE=0` → the nested lane's R0 object;
- the patched source with the switch off → the SDK's object;
- relink with the unmodified object → the staged images: mcf `fa650b98b77a1ba1`, gcc `58124bcc766ceb17`;
- relink with the R0 object → `24a4cc5ee443257e`.

| arm | object | mcf | gcc |
|---|---|---|---|
| ctl | the SDK's | `fa650b98b77a1ba1` | `58124bcc766ceb17` |
| H | HWM=1 | `60fc49da70ccf265` | `3967191163769b02` |
| HL | HWM=1 + TEST_LOG (diagnostic) | `381549aa8d114559` | `34c0d42ed8d0ddc6` |
| FL | TEST_LOG only, Full's rule (diagnostic) | `a72fe273b4d72392` | - |

## Runs

The nested lane's configuration for the mcf train runs:
- harness `nst-run-mem.sh`, copied to perf/mrhwm/run.sh;
- kernel perf/tree/images-tree, 936M;
- records 262,144, module flags as recorded for those runs, slot 29;
- one run each, under the shared qemu.lock, in this order (`perf/mrhwm/chain.sh`).

| # | run | stage |
|---|---|---|
| 1-3 | mcftest-ctl, -h, -hl | stages-frcdkmsa3/505.mcf_r-native (test input) |
| 4-6 | gcctrain01-ctl, -h, -hl | stages-frcdkmsa3/502.gcc_r-train01-native |
| 7 | mcftrain-fl | stages-frcdkmsa3-train/505.mcf_r-train-native |
| 8 | mcftrain-h | same |
| 9 | mcftrain-hl | same |

Readings:
- rc and specdiff (`validate-plan.py`);
- `@@NATIVE-CNT` begin and end: k1, k3, k4 and service_other;
- `@@MEM-PEAK` VmHWM;
- for a killed run, the kernel's OOM line;
- for logged runs, `tools/hmcheck.py` over the program's stderr. It recomputes live and cached from the events and
  checks them against the printed values; its positive and negative controls were run before this file.

## Predictions

1. **mcf test:**
   - ctl, H and HL each give specdiff 2/2;
   - H's k1 + k3 + k4 is within 5 % of ctl's. The rule keeps the 64,094,208-byte arcs_pointer_sorted cached between
     passes, so mr_make_room's gain (realloc-grow/RESULTS.md, C row) stays;
   - HL: hmcheck --hwm rc 0. The number of high-water evictions (H events) is a reading, not predicted.
2. **gcc train01:**
   - specdiff 1/1 on each arm;
   - H's service_other is at most 1.15 × ctl's (the guard of realloc-grow);
   - HL: hmcheck --hwm rc 0.
3. **mcf train FL:**
   - OOM-killed, as Full was;
   - at the last @@HM line, cached is at least 7,000,000 bytes, and live + cached exceeds peak live by at least that.
4. **mcf train H:** completes, specdiff 2/2, VmHWM in [884,180, 887,180] kB (R0 ± 1,500 kB). The live-mapping table
   (`mr_live`, up to 768 KiB of bss) is resident in H and absent in R0.
5. **mcf train HL:** completes, specdiff 2/2, hmcheck --hwm rc 0.

## Outcomes

| outcome | reading |
|---|---|
| 1-5 hold | The rule removes the OOM and keeps the cache's measured gains. It becomes a candidate configuration, Full + HWM. No board arm until Plain's `plain_reuse.c` carries the same rule, because the comparison would otherwise favour one arm. That arm is a new SPEC addendum. |
| 4 fails (H killed) | The rule is not enough. FL and HL show what stays resident. The row stays "killed by OOM". |
| 3 shows cached < 7 MB at the kill | The cache is not the whole 7 MB gap. Report what is. |
| 1 or 2 fails | The rule costs exits or reuse. Record it. A variant with slack is a new pre-registration. |
| hmcheck reports a model mismatch | The instrument is wrong. The logged runs carry no reading. |

These are twin readings, N=1, and diagnostics. The paper's Full row does not change.
