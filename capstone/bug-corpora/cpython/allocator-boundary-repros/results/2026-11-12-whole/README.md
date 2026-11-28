# 2026-11-12: all 36 cases on both virtual arms, hash seed 0, images with zlib

The first reading of cases 33-36 on the one-operand lifetime instructions, taken with the 32 cases
of the declared bundles in the same boot, so the four new rows and the 32 old ones come from one
image. The invocation is that of `../2026-11-12-seed0`: `run-virtual.py --guest-env
PYTHONHASHSEED=0 --timeout 1800`, one boot for each arm. `verdict_bundles` is unchanged and still
names `../2026-11-12-seed0`.

## Platform and images

The kit of `../2026-11-12-seed0`, equal by sha256 in every platform field of `inputs.json`: QEMU
`2d5a7107890e`, module `e47bca62addf` at its default of one context, launcher `0f8e0f903725`, kernel
`48eb725a40e4`, firmware `5b5c047a0030`, compiler `0bbb1f88bc90`, runner `e0976db1c3e8`; pinned mode,
`x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.

The images are new. They were built from this branch at 3540a2d5f172 with the command of
`../2026-11-11-t1800` (`build-virtual.sh cpython`, `CAPSTONE_SDK_CORPUS=1`,
`CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin`, `CPY_TEST_CAPI=1`), and they differ from the declared
images by what this branch adds to the port: zlib 1.3.1 cross-built into the interpreter, and the
`_capstone_nomemory` module that case 36's trigger imports.

| arm | image | declared image |
|---|---|---|
| `virtual-malloc` (`virtual-cpython`) | `eeaacee65cdbd05c722a5256b1009fd747065a22290dfe7e253092ac5c8aea46` | `6304c04f9f8e` |
| `virtual-nested-pools` (`virtual-cpython-pools`) | `d81c76849fa978c68e2d55af539bc3800a03f6cb7574283f540bcac15a5f015c` | `be62bdc9c306` |

Both boots ran 64 of 64 steps. The workload completed and both controls behaved as declared: complete
and complete on `virtual-malloc`, fault and fault on `virtual-nested-pools`. No step came near the
watchdog's limit of 2700 s.

`cases-33-36/<arm>` holds an earlier boot of the same images with `--only 33,34,35,36`. Its four rows
equal the four rows of the full run on both arms, in verdict, cause, fault function and attribution
route.

## Cases 33-36

| case | `virtual-malloc` | `virtual-nested-pools` |
|---|---|---|
| 33 gh-151295 | CAUGHT, `cause=25` in `slot_bf_getbuffer` | CAUGHT, `cause=25` in `slot_bf_getbuffer` |
| 34 gh-151403 | CAUGHT, `cause=25` in `PyOS_FSPath` | CAUGHT, `cause=25` in `PyOS_FSPath` |
| 35 gh-153539 | MISSED: reached, completed | CAUGHT, `cause=25` in `memcpy` |
| 36 gh-148395 | MISSED: reached, completed | CAUGHT, `cause=25` in `memcpy` |

Every catch is at a host-ASan fault site (`../2026-10-29-asan/sites.tsv`). The four rows equal
`../2026-10-29-virtual` on both arms, in verdict, cause, fault function and attribution route. That
run used the two-operand instructions and dev's QEMU pin of that day.

## The 32 older cases against the declared bundles

| arm | tally (36 cases) | against `../2026-11-12-seed0` (32 cases) |
|---|---|---|
| `virtual-malloc` | CAUGHT 18, MISSED 17, NO-READING 1 | 30 of 32 rows equal; 09 and 15 differ |
| `virtual-nested-pools` | CAUGHT 32, MISSED 3, NO-READING 1 | 32 of 32 rows equal |

The comparison is by verdict, reason, cause, fault function and attribution route.

The two `virtual-malloc` rows that differ are both rows the record already reads as unstable from run
to run on this arm:

- **09** reads CAUGHT here as in the declared bundle, in the same function and by the same route (the
  negative control), but with `cause=24` where the declared run had `cause=28`.
  `../2026-11-12-seed0/README.md` ("Row 09") has the declared image reading CAUGHT in 1 of 7 seeded
  runs and `NO-READING:inconclusive` in the other six.
- **15** reads MISSED here, CAUGHT (`cause=24` in `visit_decref`) in the declared bundle.
  `../2026-10-29-virtual/README.md` records the same row moving from CAUGHT to MISSED between two
  boots of one day.

A single boot cannot separate either row from the change of image, so this record does not attribute
either difference to zlib.

## What this run does not settle

- `verdict_bundles` still names the 32-case bundles of `../2026-11-12-seed0`. Cases 33-36 carry this
  reading in their status lines only, so it is not derived into `case.json` or
  `results/verdicts.tsv`.
- `cheribsd-revocation` has not run cases 33-36.
- One full boot for each arm. Rows 09 and 15 on `virtual-malloc` were not repeated on these images.
