# wmem 18-21 on the Capstone domain arms, with a temporal and a spatial control (2026-10-09)

**Build: `WM_CHUNKS=OFF`**, judged against the `sublet` arm -- the arm this corpus REQUIRES. 18-21
carry no `sublet-chunks` arm.

**Verdict: 12 of 12 arms as the completed oracles state, runner exit 0. Negative control 12/12.**

| case | upstream | access | `spatial` | `sublet` |
|---|---|---|---|---|
| 11 `http-header-map` *(temporal control)* | `3a5f82dfb5` | read | completes | **cause 24** at `0x101901534` = probe |
| 13 `http-range-cursor-past-chunk` *(spatial control)* | `0261fd7da6` | read | **cause 5** at `0x101901510` = probe | **cause 5** at `0x101901510` = probe |
| 18 `rtps-batch-sample-info-unguarded` | `716a200295` | **write** | **cause 7** at `0x101901508` = probe | **cause 7** at `0x101901508` = probe |
| 19 `ntlmssp-blob-length-before-check` | `4a4871a831` | read | **cause 5** at `0x10190148c` = probe | **cause 5** at `0x10190148c` = probe |
| 20 `proto-undecoded-bitmap-unbounded` | `ed20250c13` | **write** | **cause 7** at `0x1019014e4` = probe | **cause 7** at `0x1019014e4` = probe |
| 21 `tcp-flags-str-sixteen-bytes` | `69dac89280` | **write** | **cause 7** at `0x1019014e4` = probe | **cause 7** at `0x1019014e4` = probe |

Every fault landed on the labelled probe: the pc equals the address the boot itself published
for that probe, so the check cannot pass by a relink moving the probe.

## The first run read FAIL on all 8 new arms -- the ORACLES were incomplete, not the reading

The four cases were filed on 2026-10-08 with an oracle that said only `PREDICTED: faults`, naming
neither the probe nor the cause. The runner then fell back to its defaults -- the READ probe, and
the temporal causes 24/25 -- and scored the first run (identical images) FAIL: 18, 20 and 21
faulted with cause 7 at the WRITE probe, and 19 faulted with cause 5 at exactly the read probe but
outside the default cause pair. The oracles were then completed FROM THE SOURCE, not from the run:
the probe is the one each case.c calls (`wm_write_probe` in 18, 20, 21; `wm_probe` in 19), and the
cause follows from the access (a bounds violation on a store raises 7, on a load 5, established by
cases 16-17 on 2026-10-05). The pre-run text is kept in each arm's `prediction` field.

## The negative control had been unable to run since 2026-10-05 19:32

`--negative-control` corrupts the input record so the guest-side loader refuses it (exit 3) before
any domain exists. `refuse_if_no_result`, added at `fd75d7d89f64` an hour after this runner's last
passing negative control, treats a boot with no case marker and no fault as infrastructure and
exits 75 -- which is exactly the boot the control is built to produce, so the control stopped on
its first arm, every time. It failed LOUDLY (exit 75), and no result since then claims a wmem
negative control. Fixed: under `--negative-control` only, a visible loader exit 3 counts as the
boot having said something; any other empty boot is still refused. The guard was negative-tested
in isolation on the real refused serial (accepted only under the control) and on empty, exit-0 and
exit-30 boots (refused). Then: **12/12 oracles reported FAIL as they must**, runner exit 0.

## Infrastructure

Four boots of about 55 today ended before the guest ran anything (QEMU stopped at OpenSBI, at login,
or straight after the guest command started). Each was refused with exit 75 and re-run in a fresh
output directory; none is in the matrix.

## Inputs

- QEMU `6e5e136a070b` (the shared capstone-qemu build), the shared buildroot QEMU images.
- Domain images from `ports/wireshark/wmem` preset `capstone-domain`, `-DWM_CHUNKS=OFF`,
  `-DWM_CORPUS_DIR=<this corpus>`; compiler 7d01722aab88. Hashes in `inputs.json`.
- Loader `2187dd8f4dd6`, byte-identical to the 2026-10-05 runs' loader.

## Result lines

    OK   case=11 11-http-header-map                       spatial  completes
    OK   case=11 11-http-header-map                       sublet   cause=24 pc=0x101901534 expected=0x101901534 site=read
    OK   case=13 13-http-range-cursor-past-chunk          spatial  cause=5 pc=0x101901510 expected=0x101901510 site=read
    OK   case=13 13-http-range-cursor-past-chunk          sublet   cause=5 pc=0x101901510 expected=0x101901510 site=read
    OK   case=18 18-rtps-batch-sample-info-unguarded      spatial  cause=7 pc=0x101901508 expected=0x101901508 site=write
    OK   case=18 18-rtps-batch-sample-info-unguarded      sublet   cause=7 pc=0x101901508 expected=0x101901508 site=write
    OK   case=19 19-ntlmssp-blob-length-before-check      spatial  cause=5 pc=0x10190148c expected=0x10190148c site=read
    OK   case=19 19-ntlmssp-blob-length-before-check      sublet   cause=5 pc=0x10190148c expected=0x10190148c site=read
    OK   case=20 20-proto-undecoded-bitmap-unbounded      spatial  cause=7 pc=0x1019014e4 expected=0x1019014e4 site=write
    OK   case=20 20-proto-undecoded-bitmap-unbounded      sublet   cause=7 pc=0x1019014e4 expected=0x1019014e4 site=write
    OK   case=21 21-tcp-flags-str-sixteen-bytes           spatial  cause=7 pc=0x1019014e4 expected=0x1019014e4 site=write
    OK   case=21 21-tcp-flags-str-sixteen-bytes           sublet   cause=7 pc=0x1019014e4 expected=0x1019014e4 site=write

    negative control: 12/12 oracles reported FAIL as they must

