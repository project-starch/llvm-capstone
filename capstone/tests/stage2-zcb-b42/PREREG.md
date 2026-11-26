# Zcb with a capability base on b42c8c644: LSHOFFL against ZCBB, three boots each (pre-registration, 2026-11-26)

Stage 2's step-4 plan (plans/2026-11-25-stage2-trap-route-decoder-spec.md) names one board experiment that needs no
new seal:
- Zcb's byte and halfword forms, and its c.lbu/c.lh/c.sh family, with a capability base;
- compiler features +zcb +cap-base-zcb;
- no ISA change.

rtl reads b42c8c644 as decoding Zcb (compressed_decoder.sv:152-153), and its configuration package enables it
(capstone_cv64a6_imafdc_sv39_config_pkg.sv:77 `CVA6ConfigZcbExtEn = 1`, read here). Written before any boot of
these images.

## Arms

Both arms are built by the lane compiler: llvm-sinkland clang sha256 35580b17947b74ec, linked by its lld
(c45af5acebd278ec), with B5TL1's flags and SDK.

| arm | features beyond B5TL1's | cpugcc_r | imagick_r | perlbench_r | nab_r |
|---|---|---|---|---|---|
| LSHOFFL (off) | none | 87ef79e20e4cdf1c | a666e331f10d73cc | f3c8c80d71844564 | da57e5473a33e6fd |
| ZCBB (on) | +zcb +cap-base-zcb | 9f1a3dcbb04740bf | b69494c1d2552ded | e01764bbbe884386 | 84e2c00e7f41bbe9 |

**LSHOFFL is LSHOFF rebuilt by this compiler and linker,** byte-identical to LSHOFF on 10 of its 11 images.
- The exception is cpugcc_r. Stripped of .comment, only toplev.o differs among gcc's 387 objects. It embeds the
  compiler's revision string (__VERSION__), whose new place in the merged strings moves 66,290 bytes of references.
- LSHOFF's own objects relinked by this lld reproduce LSHOFF's gcc image exactly.
- So both arms here come from one compiler binary, and the only difference between them is the two features.

**Static text change, ZCBB against LSHOFFL:** perl -0.72 %, gcc -0.45 %, imagick -0.31 %, nab -0.46 %. These are
sizes, not fetch bytes.

## Images and platform

| image | fw_payload.bin sha256 | bytes |
|---|---|---:|
| S-LSHOFFL | 21d18193b86359a88047d30036c0ef01407fa4c0805b2e794d2a7f781040c2cb | 25,856,520 |
| S-ZCBB | bf5c9db1c13256b74b9b9e678f98c3ec3ae14e98151939fa47895a9d7ffd1fdb | 25,856,520 |

- **Built as the LSRH board form's S images** (tests/spec8-lsrh-board, Amendment 1), by the same script:
  - stages gcc, imagick (run, then val), perlbench makerand and nab, at 3,600 s each;
  - the campaign Full kernel, keep-tags firmware with FW_FAST_RDTIME, bitstream b42c8c644.
- **Checks:**
  - every non-native stage file is byte-identical to the campaign qualification overlay;
  - every native is the arm's strip.dom;
  - fast-path-refs=2;
  - the Full tree's .config, read after these relinks, carries all eleven keys that the LSRH form's Amendment 1
    lists.
- **Size.** Both are the LSRH S images' size, which runtime's C10 ruling covers (spec8-lsrh-board, Amendment 2):
  - neither boots before the LSRH S bracket's first boot has loaded at this size;
  - if that boot fails to load, this run waits for a new ruling;
  - each row records C10's BLOCK and the ruling.

## The session

- **One board, b42c8c644, no reflash, six boots:** OFF, ON, OFF, ON, OFF, ON.
  - The order alternates, so a drift through the session falls on both arms.
  - It runs after the LSRH S bracket, at about 1,050 s of stages per boot, so about 2 h.
- **Preflight runs before every boot,** and its exit line stays in that boot's run log.
- **A reflash, or another bitstream's boot, inside the six voids the session,** and the session restarts. Other
  b42c8c644 images may not run between the six.

## The reading

For each stage, t is up1 - up0 on its run line, at the 0.01 s print quantum. Each arm has three readings per stage.
mean() and range() are over those three; range is max - min.

**The binding gate, one-sided no-regression, per stage:**

    mean(ON) - mean(OFF) <= 0.0005 * mean(OFF) + range(OFF) + 0.02 s

**A gain** is claimed for a stage only if every ON reading lies below every OFF reading by more than 0.02 s, that is
max(ON) < min(OFF) - 0.02 s. Otherwise the stage reads flat or worse, as the gate says.

**UNDECIDED, not a pass,** for a stage if any of these hold:
- in any of the six boots, the stage faults, reads NO READING, is cut, or fails specdiff;
- its output hashes differ between arms;
- the drift control fires.

**The drift control** is imagick's val phase, the same plain binary in all six boots. It fires if
|mean(val, ON) - mean(val, OFF)| > range(val, OFF) + 0.02 s.

## Predictions

The prior is ZCB9 on P1 silicon (stage-2 plan, 2026-11-22), N=1. It read perl -0.33 % against a 0.889 % fetch-byte
saving, xz -0.06 % and imagick flat. The band of the stage-2 plan is time saved = fetch saved x [0, 0.37]. Its
inputs are P1's ZCB1 fetch savings; Line B's twin fetch reading is not yet taken.

| stage | P1 ZCB fetch saved | predicted | at the read price on b42c8c644 |
|---|---:|---|---|
| perlbench makerand | 0.889 % | gain or flat, at most -0.33 % | 0 to -0.91 s of 275.70 s |
| gcc t1.c | 0.593 % | flat: at most -0.22 % is 0.012 s, below the quantum | 5.55 s |
| imagick | 0.982 % | flat, as ZCB9 read on P1 | 11.6 s |
| nab | 0.220 % | flat: at most -0.08 % | 752.86 s, raw and not yet graded |

- No stage is predicted to fail the gate.
- **The perl gain is the one value this run can see.** Three boots per arm are a minimum: ZCB9's gain was 0.33 %,
  near P1's spread.
- **If Line B's twin fetch reading for ZCBB lands before the first boot of these images,** it is added by an
  amendment as the per-stage prior, with the band unchanged.

## What decides

- **All four stages pass the gate, and perl reads a gain:** ZCB is a candidate compiler row for the campaign,
  B5TL1's flags with +zcb +cap-base-zcb. It needs no bitstream. Whether it enters is the runtime lane's and the
  lead's decision.
- **All four pass with perl flat:** ZCB costs nothing measurable and gains nothing measurable on these stages. That
  is recorded as such.
- **Any stage fails the gate:** ZCB does not enter.

## What this does not decide

- CRVC, which needs a new seal.
- x264 and xz, the two stages with the largest P1 Zcb fetch savings (1.29 % and 1.26 %). x264 costs at least about
  14,300 s per boot. xz read flat on P1 silicon.
- P1.
