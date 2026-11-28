# Stage 2's x264 bracket on the seal N: X-OFF, X-ON, X-OFF2 (pre-registration, 2026-11-28)

The lead asked for more measurement before deciding on the stage-2 seal: "measure more first", after
tests/stage2-crvc-board RESULTS 30a2f27261b8 and its addenda. This bracket is x264 on N. Written before any X image
is booted on a board.

**What it cannot decide, stated first:**
- **It cannot make or unmake the GO.** x264's band is 2.162 % (its CRVC fetch saving, tests/stage2-lineb-step4) ×
  [0, 0.37], so at most 0.80 %, about 97 s of a run of about 12,200 s. That is below the 1 % a GO stage needs, and the
  GO verdict (gcc) is not reopened.
- **It cannot attribute a gain to CRVC.** No placement control exists for x264, and perlbench's placement spread was
  4.1 % (tests/stage2-placement-b42 RESULTS, c89f177d0403). A reduction it reads is "indistinguishable from
  placement", by the rule used for LSRH and ZCB.
- **It does not read the seal's cost on x264.** There is no X boot on b42c8c644 under K. The campaign's stock x264b
  rows on b42c8c644 are not a reference: they are a different image (stock kernel and firmware), and K images read
  differently from stock on perlbench (K0b and K0c lie about 0.8 s below the stock window).

**What it decides:**
1. **Whether x264 reads worse on N with CRVC,** against its own in-session boot spread.
2. **The sign and size of x264's Δ,** with r = (-Δ / mean(OFF)) / 0.02162 reported.
3. **Whether the firmware control holds** after a four-hour stage, on all three boots.

## The images

Built by c4/spec8/board/make-board-KX.sh. That is make-board-K.sh's recipe with make-board.sh's X stage: the
campaign's x264b overlay stage B5TL1-525.x264_r, bound 43,200 s, with bin/native replaced by the arm's strip.dom, then
K-crvcmis (bound 120 s), relinked by K-fpga's make-payload.sh: kernel K, the firmware clone at 514047391ee0,
FW_FAST_RDTIME=y.

| image | x264_r native | fw_payload.bin sha256 | bytes |
|---|---|---|---|
| X-LSHOFFL-K (X-OFF, X-OFF2) | LSHOFFL's, f686fe9679b04d69 (LSRH's X-LSHOFF native) | ce7c46d25831c0097351214803e6e63c7bd94fa8c047b682291858614c90a6db | 21,662,216 |
| X-CRVCBN-K (X-ON) | CRVCB's, d0dd51dd9936ade1 (flagged) | 5effe5967cdaeb723509fb583f28f7b907d23d935dc96ecdd1c4cbab8f4cdab7 | 21,662,216 |

- **Each overlay check:** 6 files equal to source, 1 native equal to the arm, 0 mismatches. keep=y,
  fast-path-refs=2.
- **The two overlays** (14 files) differ in exactly the x264 native.
- **Against LSRH's stock X-LSHOFF overlay,** X-LSHOFFL-K differs only in stages.order and stages.bound and in the two
  K-crvcmis files.
- **The size, 21,662,216 B,** equals the X images' already proven on b42c8c644. On N it is unproven, so X-OFF is its
  proof.

## The session

**boot: FLASH-N, X-OFF, X-ON, X-OFF2, FLASH-BACK**

- **The board:** the one runtime assigns, the first free after phase B.
  - FLASH-N flashes N (stage2-crvc-board Amendment 1: the seal's name and sha256 0149657ee39f6ff8…), read back.
  - FLASH-BACK flashes the board's own resident bitstream from before the session, read back.
- **No reflash** and no other bitstream's boot between X-OFF and X-OFF2. Either voids the bracket.
- **C10:** a DECLARED bake by content, under runtime's ruling. The board lane builds the declared dirs as for K, and a
  crossed control (the ON image against the OFF dirs) must BLOCK. Each size is at most 26,000,000 B.
- **X-OFF is the size proof on N.** If it does not reach /init after one retry, the bracket stops before X-ON,
  FLASH-BACK still runs, and the stop is infra.

## The reading

**The phases:** t_gen and t_run are up1 - up0 on the x264 stage's two @@SPEC-END lines, "gen" (ldecod) and "run"
(x264_r).
- **t_run is the reading.**
- **t_gen is the drift control:** the same plain ldecod binary on the same input in all three boots.

**The rules:**
- **Boot spread:** B = |t_run(X-OFF2) - t_run(X-OFF)|.
- **The difference:** Δ = t_run(X-ON) - mean(t_run(X-OFF), t_run(X-OFF2)).
- **No worse:** "x264 reads worse on N with CRVC" if Δ > 0.0005 × mean(OFF) + B + 0.02 s. Otherwise it holds.
- **A reduction:** reported when -Δ > 0.0005 × mean(OFF) + B + 0.02 s, as "a reduction indistinguishable from
  placement". It is not a gain, and not attributed to CRVC.
- **The drift control:** t_gen(X-ON) must lie within [min - q, max + q] of the two OFF gens, with q = 0.0005 × max +
  0.02 s. If it fires, the bracket is UNDECIDED.
- **UNDECIDED** also covers a fault, a cut stage, a failed specdiff, or output hashes that differ between the three
  boots.
  - **Outputs** are compared on the stage's hash block, sorted, with tick lines removed.
  - **One file is excluded: log.dec.** It is ldecod's decoder-statistics log, from the gen phase, and it records the
    date, time and per-frame milliseconds (ldecod's strings: "| ver | Date | Time |", "Time(ms)"). The two twin
    boots below differ in log.dec alone.
  - The stage's own validation is specdiff of the three imagevalidate frames, which does not read it.
- **crvcmis:** stage2-crvc-board's N predicate (Amendments 5 and 6), on all three boots.
  - Within the window: exactly one `FWCRVC redirect=1 emulate-u=M`, `@@CRVCMIS rc=135`, "binds in the CRVC mode", no
    "no trap".
  - In every boot: the banner before Linux, and no FWCRVC line inside the x264 stage.
  - It is read, never a stop.

## Predictions

1. **No worse holds.** Δ ≤ 0.0005 × mean + B + 0.02.
2. **Δ lies in [-97, 0] s,** the band's range. Two reference points:
   - at gcc's measured r = 0.179, about -47 s;
   - at perlbench's r ≈ 0, about 0.
   - **Predicted:** -Δ below 47 s, so r below gcc's.
3. **crvcmis holds on all three boots.**
4. **Outputs are equal across the three boots.**

## Limits

- No GO decision, no attribution to CRVC, and no seal cost on x264 (see the top).
- A follow-on that would attribute: a placement bracket for x264 (relinks of LSHOFFL's x264 objects, as
  stage2-placement-b42 did), about 12 h more. It is not part of this pre-registration.

## The board-form twin boots (harness tests, before this pre-registration)

Each X image's own initramfs (out/initramfs.cpio: the board init and both stages) was booted on the twin before this
was written: K-twin plain, the fw-crvc twin build, x-capstone-crvc=true, the board's bootargs, no profiler, under the
QEMU lock. These are harness tests, not measurements; twin times are not board times.

| boot | image | gen | run | x264 binds in the mode | crvcmis | log sha256 |
|---|---|---:|---:|---|---|---|
| BTX-ON | X-CRVCBN-K | 113.95 s | 4,905.34 s | yes | one `FWCRVC redirect=1 emulate-u=0`, `@@CRVCMIS rc=135`: N holds | ea9d8206a72f082f |
| BTX-OFF | X-LSHOFFL-K | 122.37 s | 4,960.02 s | no | the same: N holds | e81156f3936f8a81 |

- **Both ran to "TRAPTEST: done".** The x264 stage returned rc 0, and each boot printed the banner before Linux and
  no FWCRVC line inside the x264 stage.
- **Outputs:** the hash blocks (15 files) are equal except log.dec:
  - BuckBunny_New.264 is 0e453c44… in both;
  - frames 0, 50, 100, 150 and 155 are equal;
  - BuckBunny.yuv is equal.
- **This is the reason for the output rule's exclusion above.**

## Amendment 1 (2026-11-28, before any X board boot): x264's run on silicon is about 32,840 s, not 12,200 s; the drift control is checked against silicon gen times

**The error.** The pre-registration took x264's run as "about 12,200 s". That came from tests/spec8-lsrh-board's "a run of
more than 12,180 s", a floor read from runs that were cut. The campaign's completed x264b boots read the run phase, up1 -
up0 on its SPEC-END run line, as:

| boot | bitstream | gen | run |
|---|---|---:|---:|
| camp8-b3-20261126T2207-03-CF-B5TL1-x264b | b42c8c644 | 2,056.62 | 32,838.90 |
| camp8x-b2-20261127T0342-02-CF-B5TL1-x264b | b42c8c644 | 2,056.43 | 32,816.41 |
| camp8r2-b1-20261126T2156-22-CS-B5TL1-x264b | 14f2ef81a | 2,056.00 | 32,732.00 |
| camp8r2-b1-20261126T2156-24-CS-B5TL1-x264b | 14f2ef81a | 2,056.33 | 32,709.89 |

**What changes.** Every rule is in fractions of mean(OFF), so no rule changes. The seconds the pre-registration quoted
change with the scale:
- **The band:** 2.162 % × [0, 0.37] = [0, 0.80 %], about **[0, 263] s** of a 32,800 s run, not 97 s. It is still below
  1 %, so x264 is still no GO stage.
- **Prediction 2 becomes:** Δ lies in [-263, 0] s. At gcc's r = 0.179, about -127 s; at perlbench's r ≈ 0, about 0.
  **Predicted:** -Δ below 127 s, so r below gcc's.
- **The session is about 29 h, not 13 h:** three boots of about 34,900 s (gen + run + crvcmis), plus two flashes.
  - Each boot fits its stage bound of 43,200 s, with about 8,300 s spare.
  - The board lane's session bound is 44,400 s per boot.

**The drift control, checked against silicon before any reading.**
- **The twin cannot test it.** On the twin pair, gen read 113.95 against 122.37 s, outside the window by 8.4 s. Twin
  times come from a host-timed QEMU with no icount, and they are not board times.
- **On silicon,** gen in the configuration above read 2,056.00 to 2,056.62 s in four boots, on two boards and two
  bitstreams. Its spread of 0.62 s lies inside the rule's q = 0.0005 × 2,056 + 0.02 = 1.05 s.
- Gen in the earlier r77b3x and spb1e boots (kernel and firmware before the campaign's FW_FAST_RDTIME images) read
  2,126.10 to 2,127.40 s. That is a different configuration, and it is not used.
- **The rule is unchanged.** If the three K boots on N spread gen beyond q, the drift control fires and the bracket is
  UNDECIDED, as written.
