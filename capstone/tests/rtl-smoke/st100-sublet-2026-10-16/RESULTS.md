# speedtest1 `main --size 100`, domain against native, on the Sublet bitstream: results (2026-10-17)

Two boots on `caplifive_sublet_d93a8d66b.bit`, as pre-registered (`README.md`, lane 816de2675b21), read from the
run-scoped driver transcripts by `judge-pair.py` (private), which checks every pre-registered reading arm by arm.
Before the run it was tested both ways: it passes the pre-boot judged as a pre-boot, and fails the pre-boot judged as
the pair.

## Pre-boot (size 1), 10:55-11:13

| arm | reading |
|---|---|
| k800 | retval 4 |
| native `warm` | `111130 1e792c9d`, `SPEEDTEST1-CYCLES 2199168382 ... DROPPED 0 RC 0`, `BASELINE-WARM`, TOTAL 87.844 s |
| domain | `111130 1e792c9d`, `arena_bytes=134217728`, `SPEEDTEST1-CYCLES 2674009545 ... HEAP 134217728 DROPPED 0 RC 0`, TOTAL 106.780 s; no share-trap or NOMEM |

- Every gate reading holds. The 128 MiB region arena carves on this line.
- Cycle ratio **1.2159**, against the predicted ~1.22 (sw63's 1.2195 at the same instruction ratio).

## Pair (size 100), 11:13-19:52, one boot, 31,011 s

| arm | reading | predicted |
|---|---|---|
| k800, before and after the native | retval 4, retval 4 | 4, 4 |
| native `warm` | `23674002 573a4409`; `SPEEDTEST1-CYCLES 336,762,395,125`; TOTAL 13,469.7 s | the oracle; ~337.2 G cycles, 3.75 h |
| domain | `23674002 573a4409`; `SPEEDTEST1-CYCLES 398,552,257,537`; `HEAP 134217728 DROPPED 0 RC 0`; TOTAL 15,941.1 s | the oracle; ~398.6 G cycles, 4.43 h; no NOMEM |

- **Ratio 398,552,257,537 / 336,762,395,125 = 1.1835**, inside the pre-registered ~1.18, band 1.14-1.24. §7p (sw73, a
  different bitstream) read 1.1819.
- The two absolutes fall within 0.15 % of the predictions taken from sw73's cycles per instruction:
  - native 336.76 G against 337.2 G;
  - domain 398.55 G against 398.6 G.
- Both arms hash to the native oracle, so the ratio is of one program producing one answer.
- No share-trap, NOMEM, refusal or mcause line; one boot banner after this run's load.

**Recorded, not predicted:** the native's `BASELINE-WARM INSTRS` reads 95,124,732,502. QEMU's native count was
90,025,116,459. The board's figure counts every instruction the hart retired between the two reads, so the kernel's
work during a 3.7 h run is included; the pre-registration did not predict it.

**Not established:** N = 1 per arm, as §7p.
