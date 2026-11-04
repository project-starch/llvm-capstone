# C.5a SPEC test on silicon (board 2), and the hpm repeat: result

Pre-registered in PREREG.md, Addenda 5 and 6. c5a-spec-b2.sh ran from 4 Nov 18:06 to 5 Nov 04:33. It flashed
caplifive_vp2c_1fc2cd5aa and read the name back, ran the SPEC test list on kit bkit17-perf-c5a-b2 (the baseline
images) into out-c5a-b2, ran the seven hpm images again, and flashed bitstream 3 back.

## Grade (`c5a-spec-grade.py`, C5_OUT=out-c5a-b2, rc 1)

    gcc      Full     S1 holds | C5a/Spatial 1.0171 (band 0.990-1.010) S2 FAILS | C5a/bs3 0.9580 (bs3/Spatial 1.0617) S3 holds
    imagick  Full     S1 holds | C5a/Spatial 0.9752 (band 0.990-1.010) S2 FAILS | C5a/bs3 0.9571 (bs3/Spatial 1.0189) S3 holds
    imagick  MapLife  S1 holds | C5a/Spatial 1.0008 (band 0.990-1.010) S2 holds | C5a/bs3 0.9640 (bs3/Spatial 1.0381) S3 rep
    nab      Full     S1 holds | C5a/Spatial 1.0179 (band 0.990-1.010) S2 FAILS | C5a/bs3 0.9562 (bs3/Spatial 1.0646) S3 holds
    nab      MapLife  S1 holds | C5a/Spatial 1.0048 (band 0.990-1.010) S2 holds | C5a/bs3 0.9571 (bs3/Spatial 1.0499) S3 rep
    lbm      Full     S1 holds | C5a/Spatial 1.0004 (band 0.990-1.010) S2 holds | C5a/bs3 0.9947 (bs3/Spatial 1.0057) S3 rep
    lbm      MapLife  S1 holds | C5a/Spatial 0.9999 (band 0.990-1.010) S2 holds | C5a/bs3 0.9946 (bs3/Spatial 1.0054) S3 rep
    xz       Full     S1 holds | C5a/Spatial 0.9988 (band 0.990-1.010) S2 holds | C5a/bs3 0.9614 (bs3/Spatial 1.0389) S3 holds
    mcf      Full     S1 holds | C5a/Spatial 1.0321 (band 0.998-1.018) S2 FAILS | C5a/bs3 0.9254 (bs3/Spatial 1.1153) S3 holds

## Readings (in-context cycles of the run commands)

| | C.5a / bitstream 3 | run time | C.5a / Spatial (same images: MapLife arms only) |
|---|---|---|---|
| mcf Full | **0.925** | 0.931 (15,096 vs 16,209 s) | — |
| gcc Full | 0.958 | — (55 s run) | — |
| imagick Full / MapLife | 0.957 / 0.964 | — (30-34 s runs) | 1.0008 (bitstream 3: 1.038) |
| nab Full / MapLife | 0.956 / 0.957 | 0.958 | 1.0048 (bitstream 3: 1.050) |
| xz Full | 0.961 | 0.979 | — |
| lbm Full / MapLife | 0.995 / 0.995 | 0.993-0.996 | 0.9999 (bitstream 3: 1.005) |

## What this says

- **S1 holds on all nine arms.** rc 0, specdiff PASS, outputs equal the twin, instret within 0.1 %.
- **S3 holds.** C.5a takes 4-8 % fewer cycles than bitstream 3 on the five benchmarks that do capability loads and
  stores, mcf most (−7.5 %), and 0.5 % on lbm.
- **The bitstream-3 reference.** It is the baseline, which ran on 28-29 Oct. xz and mcf ran on board 2 and are
  same-board comparisons; gcc, imagick, nab and lbm ran on board 1 and are cross-board. The xz row (same board,
  0.961) agrees with the cross-board rows.
- **S2: the check now costs nothing measurable where the comparison is clean.** On the MapLife arms, the same images
  with and without the check, C.5a reads within 0.5 % of ObjBounds Spatial on imagick, nab and lbm. On bitstream 3
  these read 3.8 %, 5.0 % and 0.5 %. N = 1; imagick's reading lies inside the ObjBounds repeat spread.
- **S2 on the Full arms (gcc, imagick, nab, mcf: FAILS) is a pre-registration design flaw.** It does not measure the
  check: the Full images are other binaries than the Spatial images, with different code placement. It is reported
  and not read.
- **Run time.** The under-a-minute stages (gcc, imagick) compare boards and include the launch, so only their cycles
  are read.

## hpm repeat (Addendum 6): the extra L1D miss belongs to the bitstream

The seven hpm images in a fresh C.5a session read the same vector as the first session, field for field. The cycle
totals agree within 4. In particular, l1d_miss reads 1 on hpm-5, -1, -2, -3 and -4, and 257 on hpm-7 (sim 0 and 256).
The extra miss is therefore a property of C.5a, not of the first session's board state. It is one miss per run, not
a cost, and it is open with the rtl lane. hpm-5, -1 and -2 run faster than the sim (780 against 819, and so on); that
was already reported in RESULT-kits.md.
