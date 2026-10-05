# All five spatial rows measured, on both builds, with a temporal control and the negative control

**Build: `WM_CHUNKS=ON`**, judged against `sublet-chunks`. Sibling:
[`../20261005-qemu-spatial-5-OFF/`](../20261005-qemu-spatial-5-OFF/README.md), the region-granular
build.

**Verdict: 12 of 12 arms as predicted, runner exit 0 on EACH build. Negative control 12/12.**

| case | upstream | crossing | `spatial` | protected arm |
|---|---|---|---|---|
| 11 `http-header-map` *(temporal control)* | `3a5f82dfb5` | — | completes | **cause 24** |
| 13 `http-range-cursor-past-chunk` | `0261fd7da6` | read past a chunk | **cause 5** | **cause 5** |
| 14 `solaredge-payload-six-past` | `1d8acb21ab` | read six bytes past | **cause 5** | **cause 5** |
| 15 `opcua-padding-below-chunk` | `d24613c461` | read **below** a chunk | **cause 5** | **cause 5** |
| 16 `dcp-etsi-rs-parity-write` | `e8ef9df09d` | **write** past a buffer | **cause 7** | **cause 7** |
| 17 `dns-one-byte-write` | `5a560f3f6a` | **one-byte write** past | **cause 7** | **cause 7** |

Every fault landed on the labelled probe with the pc equal to the address the run resolved from the
image. Cases 14 and 15 are the corpus's only rows **live at the `v4.6.8` pin**; 13, 16 and 17 are
fix-reversals.

## A pre-registered prediction was refuted: loads fault 5, stores fault 7

All five rows were pre-registered with **cause 5**. The three read rows read 5; **both write rows
read 7** — a bounds violation on a *store* is a different cause from one on a *load*. The
pre-registration was right about the probe and the arm and wrong about load-versus-store, and the
`case.json` files for 16 and 17 record the refutation rather than quietly carrying the corrected
number.

## Both builds fault, which is the finding and not a disappointment

`spatial` and the protected arm read the same on every spatial row, on **both** builds, because
`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15` (`wm_narrow()`) narrows **every** wmem
allocation to its request under `WM_DOMAIN`, whatever `WM_CHUNKS` is set to. So this harness has no
malloc-granular arm to contrast against, and these rows do **not** discriminate the chunk port.
Case 13's run established that by refuting its own prediction; these five were pre-registered with
the correction applied.

**The contrast lives in the tshark app port**, whose measured fx12 length ladder is `level0`
41 908 912, `shrink` 8 388 560, `sublet` 1 048 528, `chunks` 64.

## The negative control, which is what makes 12/12 mean anything

`--negative-control` over the same six cases and both modes: **12/12 oracles reported FAIL as they
must**, exit 0. Every oracle in this run — including all ten new ones — is proven able to fail.

## One infrastructure failure, recorded because it was not a result

In the first pass, case 15's `sublet` arm on this build produced **no output at all**: its serial
capture ends at `sh /mnt/host/run.sh` with no marker, no fault and no `CONTROL-FAILED`. The runner
reported it as a FAIL rather than exiting 75, which is a weakness in `run-defects.py` worth knowing
about — the memcached runner has a "BOOT PRODUCED NO RESULT" check and this one does not. Re-run
alone, the arm passes, and it passes in this bundle. **A boot that produces nothing is not a
measurement.**

QEMU only. **N = 1 per cell.** Files: `matrix.tsv`, `inputs.json`.
