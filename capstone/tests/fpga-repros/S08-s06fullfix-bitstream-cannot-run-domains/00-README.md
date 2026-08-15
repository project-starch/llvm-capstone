# S-08 — on `caplifive_s06fullfix.bit` the monitor takes an UNHANDLED TRAP just after a domain's first share returns

**Status: OPEN, BLOCKING all board work. Reported 2026-08-15, immediately after the reflash.**

**This may not be a bitstream defect.** The bitstream is demonstrably the variable, but that does not
distinguish two readings, and the second is at least as likely:

1. the new RTL is wrong; **or**
2. **the new RTL is right and our MONITOR now needs a matching change.** The S-06 work includes
   *P5a: stored tags for capability CSRs, and CPMP grants require the tag*, and *P5b: the
   dom-switcher carries a real 129-bit tag lane*. Our monitor installs region capabilities through
   `read_cpmp`/`write_cpmp` (`sbi_capstone.c:333+`), and the failure lands **immediately after the
   dom-switcher has run**. A CPMP grant that is no longer honoured because it lacks a tag would
   produce exactly this.

Please read (2) before assuming (1). We cannot tell them apart from the board side.

Sibling issues, if one of those is your symptom: `S07-capability-untagged-on-reload/`,
`S06-untagged-ldc-stc-high-half/`, `S01-image-perturbation-hang/`.

---

## The discriminator: `EXCX` fires 4/4 on the new bitstream and 0/14 on the old, on the SAME firmware

`EXCX:0000E002` is emitted from the unconditional `default:` arm of the monitor's `handle_exception`
(`capstone/caplifive-system/.../capstone-sbi/sbi_capstone.c:1486-1492`) — it means **the monitor took
a trap it does not handle at all**.

| | boots | `EXCX:0000E002` |
|---|---|---|
| `caplifive_12august.bit` | 14 | **0** |
| `caplifive_s06fullfix.bit` | 4 | **4** |

The instrument is unconditionally compiled and is present in the *same firmware bytes* on both sides:
`fw_93aa9a2426bc.bin` produced zero `EXCX` across four old-bitstream boots and one `EXCX` on the new
bitstream. Counted with `python3`, not `grep` (which returns empty on these logs).

**This is the whole case. Everything below is corroboration.**

## The domain DOES run — an earlier draft of this file said it did not

The capability machinery works. From the new-bitstream transcript, the domain is created, two regions
are made and mapped, and a full annotated share completes, including the monitor's entry into and
return from the domain:

```
SQ: A/dom-ok ... SQ: B/mkregion1 ... SQ: C/mkregion2 ... SQ: D/mapped ... SQ: E/share1
SHA0..SHA4 ... SHA5:00000000  SHA6:00000000  ECSZ:00000000
EXCX:0000E002  MCAU:00000008  MEPC:89FBCB54  MTVL:00000073
```

`SHA5` = about to leave M-mode for the domain; `SHA6` = the domain returned; `ECSZ` = the handler
returned (`sbi_capstone.c:119,125,126`). This is byte-for-byte the old-bitstream success sequence up
to `ECSZ`. **The failure is a trap the monitor cannot handle, taken at the first instant after the
first annotated share returns** — not an inability to run domains.

## The two observed signatures are ONE shape

| driver | control | `MCAU` | note |
|---|---|---|---|
| baked rungs | `k800` | `0000000C` (fetch page fault) | ×2 boots, identical **latched trap state** (the transcripts differ: `BASE` moves) |
| SQLite stages | `L2` | `00000008` | ×2 boots, one on new firmware, one on the byte-identical old image |

Both are `EXCX:0000E002`, same emit site, both immediately after `ECSZ`. They differ only in the
cause code. Note `handle_exception` services `CAUSE_FETCH_ACCESS` (1) via `swap_cpmp` but sends
**12** to the unhandled `default:` — so a CPMP denial arriving with a different cause code produces
the `k800` signature exactly.

The cause-8 case, offered as an observation and not a mechanism: `mepc` `0x3f9b0a3b54` /
`0x3f89fbcb54` — identical low 12 bits, ASLR-shifted middle — with `mtval = 0x73`, the `ecall`
encoding. That is a **U-mode ecall at a host-range address reaching M-mode's unhandled path**, on
firmware whose dispatch never sent one there on the old bitstream.

## Corrections to an earlier version of this file, listed so nothing is taken on trust

* **`MEPC:800072D0` is NOT a new-bitstream fingerprint.** `0xffffffff800072cc` — four bytes away, and
  the `k800` value is a 32-bit truncated print of what is very likely the same site — appears as the
  latched mepc in **9 of the OLD bitstream's driver logs** (`restore1/2`, `timemachine1`, `boot2`,
  `pair1`, `patch1-4`, `probe3`). Anyone grepping an old log would rightly close this as
  pre-existing. Dropped as evidence.
* **"Byte-identical boots"** applies to the latched trap state only; the transcripts differ.
* **"Control entered 14/14 across two drivers over four days"** is one driver (the SQLite stages
  path) over two days. The baked-rungs path has **no surviving old-bitstream control at all**, and
  its oracle had to be regenerated today, so `k800` is corroborating shape, not independent evidence.
* **Firmware exclusion rests on one A/B boot**, not four. Boots 1-3 used a firmware built after the
  reflash that never ran on the old bitstream. What carries the exclusion is the `EXCX` identity
  *across both firmwares*.
* **The driver printed "THIS RUN CARRIES NO VERDICT" and we overrode it deliberately.** The override
  is justified because the classifier is provably wrong here: its canned text asserts "markers stop
  at SHA5 with no SHA6" while `SHA6:00000000` is in the same transcript, and its monitor-wedge regex
  (`run_sqlite_stages_fpga.py:474`) matches only `SPL[AB]|ILLX`, so `EXCX` — the one tag that means
  an M-mode monitor wedge — was never considered. Disclosed rather than silently relied upon.

## What is NOT established

* **The mechanism.** We do not know what the monitor did to earn an unhandled trap.
* **Defect vs required-change**, as above.
* **The S-06 fix is UNTESTED, not failing.** No boot with a passing control has occurred on this
  bitstream, so `s06agg` has never been read on it. There is no acceptance verdict in either
  direction.
* **No pre-reflash baseline exists.** The boot intended to record the outgoing bitstream's S-06
  values aborted on a missing oracle file before it ran, and the reflash came first.

## Reproducing

```
cd capstone/tests/rtl-smoke
export FPGA_URL=<FPGA-CONSOLE-URL>
export FPGA_FW=<...>/fw_payload.bin
export FPGA_IMG_NAME=fw_93aa9a2426bc.bin     # the known-good stored image; skips the upload
export SQLITE_STAGE_DOMS="/test-domains/L2.dom,/test-domains/G6.dom,/test-domains/G6.dom,/test-domains/G6.dom"
python3 -m fpga_driver.run_sqlite_stages_fpga
```

`L2.dom` is a truncation arm that enters and returns; it did so on 14 of 14 old-bitstream boots.
Look for `EXCX:0000E002` in the run-scoped transcript — that is the signal.
