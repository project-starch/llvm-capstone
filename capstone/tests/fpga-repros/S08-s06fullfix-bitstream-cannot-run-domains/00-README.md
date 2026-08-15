# S-08 — `caplifive_s06fullfix.bit` cannot run any capability domain

**Status: OPEN, BLOCKING. Nothing can be measured on this bitstream.** Reported 2026-08-15,
immediately after the reflash, before any other work was attempted on it.

Sibling issues, in case one of those is your symptom: `S07-capability-untagged-on-reload/`,
`S06-untagged-ldc-stc-high-half/`, `S01-image-perturbation-hang/`.

## The observation

Four consecutive boots on `caplifive_s06fullfix.bit`, every one VOID at its **control** arm — the
first domain of the boot, which exists precisely so a failure can be attributed. Two distinct
signatures:

| boot | driver | control | latched trap state |
|---|---|---|---|
| 1 | baked rungs | `k800` returned nothing (oracle 4) | `MCAU:0000000C` = INSTR_PAGE_FAULT, `MEPC:800072D0` |
| 2 | baked rungs | identical | **byte-identical** to boot 1 |
| 3 | SQLite stages | `L2` never entered | `MCAU:00000008`, `MEPC:89FBCB54` |
| 4 | SQLite stages | `L2` never entered | `MCAU:00000008`, `MEPC:9B0A3B54` |

`MEPC:800072D0` symbolises to **`sanitize_domain+0x188`** inside the monitor. (Symbolised against
`fw_jump.elf`; the board runs `fw_payload.bin`, a different link product, so treat the symbol as
indicative and the address as exact.) Boots 1 and 2 producing *byte-identical* latched state is the
signature of a deterministic fault, not a flake.

## Why this is the bitstream and not our software

The control arm entered on **14 of 14** boots on the previous bitstream (`caplifive_12august.bit`),
across two drivers, four days, and several different firmware builds. It has now failed **4 of 4**.

The decisive A/B removes the firmware as a variable. `fw_93aa9a2426bc.bin` — the exact image, byte
for byte, recovered from the console's content-addressed image store — booted and ran domains
repeatedly on the old bitstream, including earlier the same day. Booted on the new bitstream via
`FPGA_IMG_NAME=fw_93aa9a2426bc.bin`, its control **stalls**.

**Same firmware bytes. Old bitstream: works. New bitstream: fails.** The only variable is the
bitstream.

## What has NOT been established

* **We do not know the mechanism.** An instruction page fault inside the monitor and a
  domain-never-entered stall may be one cause or two.
* **We could not take a pre-reflash baseline.** The boot intended to record the S-06 acceptance
  values on the outgoing bitstream aborted on a missing oracle file before it ran, and the reflash
  happened before it could be repeated. The reference values for broken silicon
  (`s06agg`=5, `s06aggcap`=7, `s06aggwide`=237) therefore come from earlier records, not from the
  immediately preceding bitstream.
* **The S-06 fix is UNTESTED.** No boot with a passing control has occurred on this bitstream, so
  `s06agg` has never been read on it. Whether the S-06 RTL fix works is **unknown**, not "failing".

## What we need

Either a corrected bitstream, or confirmation that this one needs a matching firmware/monitor change
we have not made. Everything downstream is blocked: the S-06 acceptance gate, SQLite re-validation,
and all S-07 work.

## Reproducing

```
cd capstone/tests/rtl-smoke
export FPGA_URL=<FPGA-CONSOLE-URL>
export FPGA_FW=<...>/fw_payload.bin
export FPGA_IMG_NAME=fw_93aa9a2426bc.bin        # the known-good stored image; skips the upload
export SQLITE_STAGE_DOMS="/test-domains/L2.dom,/test-domains/G6.dom"
python3 -m fpga_driver.run_sqlite_stages_fpga
```

`L2.dom` is a truncation arm that enters and returns; on the old bitstream it did so every time.
