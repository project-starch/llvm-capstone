# Provenance for the 2026-10-11 CheriBSD run

Result lines only — the raw serial capture is deliberately not committed. A boot
log is contaminated by construction (kernel and driver banners carry account
names and host strings), so scrubbing it is endless and per-log; the classified
rows and the measurement lines are both smaller and better evidence.

| file | what |
|---|---|
| `verdicts-20261011.txt` | `classify.py` output: per row, per arm, all three configs |
| `measurements-20261011.txt` | the `ROW`/`MEAS`/`KNOB`/`STATUS` lines the verdicts were computed from |

Vehicle, and the four runs behind this:

| run | change | why |
|---|---|---|
| 1 | first attempt | kernel panicked at boot: Sstc (`stimecmp`) vs bbl not setting `menvcfg.STCE` |
| 2 | `-cpu rv64,sstc=false` | booted; full 24-row result, all controls firing |
| 3 | + `z1_bounds_probe`, `z2_libc_uaf` | `z2` settled that revocation runs only under eager; `z1` truncated at 160 chars |
| 4 | probe prints raw capability fields; `MEAS` lines untruncated; `a5` link tags | final: bounds measured, `a5` tag hypothesis refuted |

Reproduce with `run-php-blind.sh`; see `VEHICLE-BRINGUP.md` for the vehicle and
`RESULTS.md` for the analysis.
