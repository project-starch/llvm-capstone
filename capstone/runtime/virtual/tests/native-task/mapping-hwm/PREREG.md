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

## Amendment 1 (2026-11-22 07:19, during run 8, before run 9 and before any rerun)

Runs 3, 6 and 7 (the logged images HL and FL) stopped within 16 s with SIGSEGV. The cause is the instrument. The
image's first `@@HM` write came before musl had set up the thread pointer: a cause 24 SHRINK fault in
`__capstone_delegate_call` at the inlined `dl_transport()`, whose TLS load is scheduled ahead of its `tp &&` check. The
ctl and H images never write before that point. These three runs carry no reading. Their HMCHECK lines read NO DATA
(rc 2), as the check is built to.

The fix is in `mr_log` only, under `CAPSTONE_HEAP_MAPPING_REUSE_TEST_LOG`:
- lines are queued in a 64 KiB static buffer and written once `tp` is tagged;
- a write that does not take every byte leaves them queued;
- a full queue counts the dropped lines and prints `@@HM-LOST <n>`. `hmcheck.py` exits 3 on that line; its positive
  control is `t-lost`.

The objects are unchanged without TEST_LOG. A rebuild of the switch-off object is byte-identical to the SDK's, the H
object to the measured H object, and the H relink gives `60fc49da70ccf265` again. The logged images carry 64 KiB more
bss, which is negligible against FL's predicted 7 MB gap.

| arm | mcf | gcc |
|---|---|---|
| HL (fixed) | `90096bb08f1827b7` | `f6efc0b2a91a4ef9` |
| FL (fixed) | `4fa4f0a37aa732bd` | - |

Deviation: run 9 (mcftrain-hl) has not started. Its stage image is replaced by the fixed HL image before it does. After
the chain ends, `tools/chain2.sh` reruns mcftest-hl, gcctrain01-hl and mcftrain-fl with the fixed images, in that order.
Their predictions are unchanged (1, 2 and 3 for the logged arms).

Already read before this amendment, so it changes none of their predictions:
- runs 1, 2, 4 and 5;
- run 8 up to 123 s.

## Outcome of pre-registration 1 (read 2026-11-22 09:03-09:07)

All runs are on the twin, N=1. Raw lines are in `perf/mrhwm/RESULTS`.

| # | prediction | reading | verdict |
|---|---|---|---|
| 1 | mcf test: H's k1+k3+k4 within 5 % of ctl; HL hmcheck rc 0 | 33,794 vs 32,036 (+5.49 %); HL rc 0 (627 events) | **fails** |
| 2 | gcc train01: H's service_other at most 1.15 x ctl; HL rc 0 | 13,702 vs 2,303 (5.95 x); HL rc 0 (52,236 events) | **fails** |
| 3 | mcf train FL: OOM-killed; at the last @@HM line cached >= 7,000,000 B and live + cached above peak by that | killed at 99.2 s (anon-rss 892,732 kB); cached 65,060,864 B, excess 64,303,104 B | holds |
| 4 | mcf train H: completes, specdiff 2/2, VmHWM in [884,180, 887,180] kB | rc 0, 2/2, 886,072 kB | holds |
| 5 | mcf train HL: completes, 2/2, hmcheck rc 0 | rc 0, 2/2, rc 0 (1,025 events) | holds |

Every specdiff in the table passed, and every hmcheck found 0 model mismatches and no @@HM-LOST. The rule removes the
OOM but costs reuse, so it is not a candidate as written. Following the outcome table, a variant with slack needs this
new pre-registration.

**Why the rule costs reuse.** While the heap grows, live + new exceeds peak, so the cap is live + new and the rule
empties the cache. gcc grows for most of its run.

**The replay model** (`tools/replay.py`). It replays each logged request stream through a policy; the stream does not
depend on the policy. Its positive control reproduces the logged policy exactly:
- gcc HL: hits 16,244, unmaps 5,913;
- mcf test HL: hits 203, unmaps 71;
- mcf train HL: hits 363, unmaps 97;
- mcf train FL under Full's rule: hits 40, unmaps 5.
On the two gcc readings, service_other equals fresh mappings + unmaps + 111: ctl 2,192 + 111 = 2,303 (read 2,303),
H 13,595 + 111 = 13,706 (read 13,702).

## Pre-registration 2 (written 2026-11-22 09:07, before any of its runs): the rule with 16 MiB of slack (H16)

`CAPSTONE_HEAP_MAPPING_REUSE_HWM_SLACK`: before a fresh mapping, the cache gives mappings back until live + cached + new
is at most max(peak, live + new) + slack. Default 0. Built at 0, the objects are byte-identical to the pre-registration-1
objects (off to the SDK's, H, HL).

**Choice of 16 MiB, from the replay of the three streams:**

| slack (MiB) | gcc service_other (model) | mcf train excess (MiB of mappings) |
|---|---|---|
| 8 | 2,842 (1.23 x) | 0.2 |
| 12 | 2,656 (1.15 x) | 0.2 |
| 16 | 2,532 (1.10 x) | 0.2 |
| 20 | 2,474 (1.07 x) | 0.2 |
| 24 | 2,431 (1.06 x) | 22.1 |

16 is inside the window from 12 (gcc's guard) to 24 (mcf train's cliff). It is fitted on these streams, so five
benchmarks it was not fitted on are held out.

Images use pre-registration 1's recipe. Every ctl relink reproduces its staged image: xz 191cde31, imagick 43f3fccf,
nab 8e07b37e, lbm 690e11ff.

| arm | mcf | gcc | xz | imagick | nab | lbm |
|---|---|---|---|---|---|---|
| H16 | `62690f60eb3f9429` | `d3e34e859a190019` | `3d961ae1848d126f` | `ab847ad67225aa9b` | `9a67333f83085f7a` | `f0ebc979aab2059f` |
| HL16 (diagnostic) | `bf905e3acf29c0bb` | `920b2812558e6740` | - | - | - | - |

**Runs** (`tools/chain3.sh`), one at a time, in this order:
1. gcctrain01-h16, mcftest-h16, mcftrain-h16, gcctrain01-hl16, mcftrain-hl16;
2. held out: lbm, nab, imagick and xz test, then gcc test, each as ctl then H16;
3. noise: mcftest-ctl2 and gcctrain01-ctl2, a second run of the pre-registration-1 controls.

**Predictions:**
1. mcf test H16: specdiff 2/2; k1+k3+k4 within 2 % of ctl's 32,036. The model gives H16 the same events as ctl
   (hits 242, fresh 38, unmaps 32).
2. gcc train01 H16: specdiff 1/1; service_other in [2,494, 2,570], the model's 2,532 +/- 1.5 %, which is below
   1.15 x ctl; VmHWM at most 218,000 kB (H 200,168 plus 16 MiB and noise).
3. mcf train H16: completes, specdiff 2/2, VmHWM in [884,180, 887,180] kB.
4. HL16 gcc and mcf train: specdiff pass; `hmcheck --hwm --slack 16777216` rc 0; `replay.py --check hwm+16` MATCH.
5. Held out (lbm, nab, imagick, xz, gcc test): specdiff pass on both arms; for each, H16's service_other at most
   1.15 x ctl's, and k1+k3+k4 within 5 % of ctl's.
6. Noise (a reading, not a prediction): ctl2 against ctl for k1+k3+k4 and service_other. If either moves by more than
   1 %, prediction 1 is read as unresolved rather than as held or failed.

**Outcomes:**

| outcome | next step |
|---|---|
| 1-5 hold | Full + H16 is a candidate configuration. Plain's `plain_reuse.c` gets the same rule before any board arm, under a new SPEC addendum. |
| 3 fails | Mapping bytes do not model the resident footprint; read HL16 and the OOM line. |
| 2 or 4 fails | The replay model is wrong for this rule. |
| 5 fails | 16 MiB is fitted to the three streams; record which benchmark breaks. |

## Outcome of pre-registration 2 (read 2026-11-22 12:01)

All 17 runs completed, chain3 at 1ec86beb6ec4, one at a time on the twin. Readings come from each run's
`@@NATIVE-CNT end` lines (summed over invocations) and its `@@MEM-PEAK` (`out/<run>.log`), and are recomputed by
`tools/readings.py`.

| # | Prediction | Reading | |
|---|---|---|---|
| 1 | mcf test H16: specdiff 2/2; k1+k3+k4 within 2 % of 32,036 | 2/2; 31,946 (-0.28 %) | held |
| 2 | gcc train01 H16: specdiff 1/1; service_other in [2,494, 2,570]; VmHWM at most 218,000 kB | 1/1; 2,523 (ctl 2,303: 1.096 x); 217,776 kB (ctl 245,088) | held |
| 3 | mcf train H16: completes, 2/2, VmHWM in [884,180, 887,180] kB | completes, 2/2, 886,368 kB | held |
| 4 | HL16 gcc and mcf train: specdiff pass; hmcheck --slack 16 MiB rc 0; replay --check hwm+16 MATCH | gcc 1/1, rc 0, MATCH (hits 21,755, unmaps 216); mcf 2/2, rc 0, MATCH (hits 417, unmaps 44) | held |
| 5 | held out: specdiff pass both arms; service_other at most 1.15 x ctl; k1+k3+k4 within 5 % | every arm passes; service_other equal to ctl on all five; k1+k3+k4 -0.05 % lbm, 0 nab, -3.5 % imagick, +0.01 % xz, -0.17 % gcc test | held |
| 6 | noise (a reading): ctl2 vs ctl | mcf test k1+k3+k4 +0.14 %, service_other 0; gcc train01 -0.10 %, service_other 0 | under 1 %: prediction 1 resolved |

Held-out detail:

| bench | k1+k3+k4 ctl / H16 | service_other ctl / H16 | specdiff |
|---|---|---|---|
| lbm test | 3,837 / 3,835 | 217 / 217 | 1/1, 1/1 |
| nab test | 1,666 / 1,666 | 52 / 52 | 1/1, 1/1 |
| imagick test | 846 / 816 | 80 / 80 | 1/1, 1/1 |
| xz test | 318,395 / 318,417 | 24,229 / 24,229 | 12/12, 12/12 |
| gcc test | 1,165 / 1,163 | 153 / 153 | 1/1, 1/1 |

What the reading does not say:
- The held-out five never trip the rule: service_other is identical on each, so they show that H16 costs them
  nothing, not that 16 MiB generalises to a program whose heap grows the way gcc's does.
- The imagick and gcc test runs are too short for the memory sampler (VmHWM NONE on both arms).
- Counts are twin counts of native-route events. Time on the board is not measured here.
- gcc train01's k1+k3+k4, which was not predicted, reads 7,235 against ctl's 6,961 (+3.9 %), against the noise pair's
  -0.10 %. Not attributed here; it is the reading to look at before any board arm.
- N = 1 per arm. The one noise pair per fitted program bounds run-to-run movement for those two programs only.

**Next (the table's first row):** Full + H16 is a candidate configuration. Before any board arm, Plain's riscv-base
musl `plain_reuse.c` gets the same rule with the same slack, under a new SPEC addendum.

## Correction to the outcome of pre-registration 2 (2026-11-22 12:23): gcc train01's k1+k3+k4 is attributed

The outcome read gcc train01's k1+k3+k4 at +3.9 % and left it unattributed. Split by counter (ctl / ctl2 / H16):
- k1 1,981 / 1,982 / 1,970;
- k3 2,459 / 2,459 / 2,679;
- k4 2,521 / 2,513 / 2,586;
- service_other 2,303 / 2,303 / 2,523.

k3 rises by 220, exactly service_other's rise. mcf test shows the same identity in the other direction: k3 774 to 728
and service_other 621 to 575, both -46. So 220 of the +274 are the service calls that prediction 2 already priced in.
The remainder is k4, +65 (+2.6 %; the noise pair moves it by 8).

## Pre-registration 3, part A (2026-11-22 12:23, before any of its runs): calibration of the Plain counterpart

Under Addendum 87 both SPEC arms run the same allocator. The paper's Plain images, the al3 stages, carry
`plain_reuse.c`: `__plain_map` is in the mcf image 2d48ad97ae6009c7. So H16 reaches a board arm only together with
the same rule in Plain.

`plain/plain_reuse.c` and `plain/plain-hwm.diff` (against perf/alean/plain/musl-src) add three things:
- `PLAIN_REUSE_HWM_SLACK`: the rule, with Full's pick order;
- `PLAIN_REUSE_TEST_LOG`: Full's @@HM lines;
- a realloc hook, `__plain_remapped`. Plain's realloc grows a large mapping with the kernel's mremap, outside the
  cache, and without the hook live would undercount that growth.
`hmcheck.py` and `replay.py` read the remap event (X); on Full's logs their output is unchanged.

Built with neither macro (b-off), every one of the 1,352 objects and libc.a are byte-identical to
`musl-rv64gc-alean`. With the rule on, exactly plain_reuse.lo and realloc.lo differ. Relinked with that libc
(`plain/relink.sh`, the SPEC build's recorded link line), ctl reproduces the stage images exactly: mcf 2d48ad97ae6009c7,
gcc 93bc8e196b036d07.

Images:

| variant | mcf | gcc |
|---|---|---|
| H16 | 2143e828aef3665c | 3768a3738d47f095 |
| HL0 (log, policy unchanged) | c1edb491ac75758b | cd68517252a929ad |
| HL16 | 98eeec439bfd2d10 | ca13978dc98d6db8 |

**Part A runs** (`plain/calib.sh`): ctl and HL0 of mcf test, gcc train01 and mcf train on the twin. They measure no
rule. They give Plain's own request streams, and part B's predictions (H16, HL16, held-out programs) are computed from
those streams with `replay.py` and written before any H16 run.

**Part A predictions:**
1. Every run specdiff PASS.
2. hmcheck over each HL0 log exits 0. HL0's VmHWM is within 1 % of ctl's on mcf train and gcc train01.
3. `replay.py --check ctl` over each HL0 log MATCHes. Plain's cache and Full's ctl policy are the same rule (512
   entries, 64 MiB, front eviction), and the model's hit pick, the newest match, is Plain's own scan order.
   A mismatch means the model must change before part B.

**Amendment to part A (2026-11-22 12:26, before any part A run).** The first launch ran nothing. All six runs stopped at the twin
runner's first check: it asserts the native board stage's cleanup line, and the Plain al3 board stages have their own
format. Those six are void (RESULTS). The stages are now the RVAL twins' method (perf/alean/plain/st-gcc): Full's
native-format twin stages (mrhwm/stages mcftest-ctl, gcctrain01-ctl, mcftrain-h) with only the native executable
replaced by the Plain image and plan.json's hash updated (`plain/mkstage.py`). The Plain binary has no Capstone
sections and runs as an ordinary process. ctl's executables are the board images (mcf 2d48ad97ae6009c7, gcc
93bc8e196b036d07). The predictions stand.

## Outcome of pre-registration 3 part A (read 2026-11-22 17:07)

All six calibration runs completed on the twin. The first launch was void (amendment above). In the relaunch every
run waited on the shared QEMU lock, and they ran between 14:26 and 17:06.

| # | Prediction | Reading | |
|---|---|---|---|
| 1 | specdiff PASS everywhere | mcf test 2/2 and 2/2, gcc train01 1/1 and 1/1, mcf train 2/2 and 2/2 | held |
| 2 | hmcheck rc 0 on each HL0 log; HL0 VmHWM within 1 % of ctl on mcf train and gcc train01 | hmcheck rc 0 on all three, 0 model mismatches. VmHWM: mcf train 619,896 against 619,900 kB; gcc train01 137,444 against 128,496 kB (+7.0 %) | **failed on gcc's VmHWM** |
| 3 | replay --check ctl MATCH | MATCH on all three: hits / unmaps 196/5, 16,869/0, 316/7 | held |

- The remap event is real. X appears 4 times on mcf (both inputs) and 2 times on gcc. Without the hook, live would
  have undercounted those resizes.
- **gcc's VmHWM is unexplained.** HL0 and ctl run the same policy, and hmcheck and replay show the same events, so
  the allocator is not the cause.
  - A candidate is the sampler. It reads VmHWM once a second, and Plain gcc train01 runs for about two minutes on the
    twin. If RSS is still growing when the last sample is taken, ctl can be read below its own peak.
  - That is a hypothesis, untested. Part B adds a second ctl of gcc train01 and of mcf train as noise pairs, and gcc's
    VmHWM is reported in part B as a reading only.

## Pre-registration 3 part B (2026-11-22 17:07, before any of its runs): Plain H16 on the twin

Runs (`plain/partb.sh`): H16 and HL16 of gcc train01, mcf test and mcf train, then ctl2 of gcc train01 and of mcf
train. The stages are part A's; ctl2 uses the board images. Images: H16 mcf 2143e828aef3665c, gcc 3768a3738d47f095;
HL16 mcf 98eeec439bfd2d10, gcc ca13978dc98d6db8.

Model (replay of each HL0 stream, ctl → hwm+16):

| program | hits | fresh | unmaps | live + cached peak (MiB) |
|---|---|---|---|---|
| gcc train01 | 16,869 → 16,840 | 888 → 917 | 0 → 94 | 131.2 → 109.0 |
| mcf test | 196 → 197 | 30 → 29 | 5 → 4 | 657.2 → 611.5 |
| mcf train | 316 → 317 | 38 → 37 | 7 → 5 | 673.9 → 628.2 |

**Predictions:**
1. specdiff PASS on every run.
2. Each HL16 log satisfies `hmcheck --hwm --slack 16777216` rc 0 and `replay --check hwm+16` MATCH.
3. Each HL16 log's own counts equal the model column above, which was computed from a different run's stream: U
   (hits) 16,840 / 197 / 317 and E+R+H+F (unmaps) 94 / 4 / 5. This tests that Plain's request stream does not depend
   on the policy, as Full's does.
4. H16's VmHWM is at most ctl's plus 0.1 % on mcf test and mcf train (306,012 and 619,900 kB): the rule never raises
   the footprint. No lower bound is predicted, because VmHWM counts touched pages and the mapping-byte drop
   (45.7 MiB) does not translate into resident pages by any model here.
5. Noise (a reading): ctl2 against ctl VmHWM on gcc train01 and mcf train. If gcc's ctl2 moves by more than 1 %, part
   A's gcc failure is read as instrument resolution, and gcc's VmHWM decides nothing in part B.

**Outcomes:**
- If 1-4 hold, the Plain counterpart is ready. Full + H16 and Plain + H16 then go to the board together, under a new
  SPEC addendum on mcf train (Full was killed at 13,777 s, Addendum 212) and the rows that use the cache. Board time
  is the measurement the twin cannot give.
- If 3 fails, the Plain stream depends on the policy (the remap path is the first suspect), and the model is not used
  for Plain.
- If 4 fails, the rule raises Plain's footprint, which contradicts its design. Read the HL16 log.
