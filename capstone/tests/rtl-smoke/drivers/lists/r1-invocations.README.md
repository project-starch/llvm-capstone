# `r1-invocations.txt` — the R1 release-cost campaign (90 invocations over 8 boots)

Consumed by `board-r1e4.sh`, which takes **lines `12*(k-1)+1 .. 12*k` for `R1_BOOT=k`**. Line format
is `rep arm series pattern arena`; the driver appends `--reps 1`, `--tables 65536` and `--testset r1`
itself, and brackets each boot with the `k800` control.

## What the 90 are

18 `(arm, series, pattern)` combinations x 5 repetitions. A repetition is a **fresh domain**
(METHODS), which is why each is its own invocation rather than `--reps 5`.

    arm      S (sublet) and P (spatial)                       -- the protocol's paired arms
    series   nodes, bytes, heap, depth   x  patterns shared, combined   = 8 per arm
             object                      x  pattern  individual        = 1 per arm
                                                                    9 per arm, 18 in all

Arenas: **4 MiB** for nodes/bytes/depth, **8 MiB** for heap (U reaches 4096 KiB), **2 MiB** for object.

## Why the order is rep-major, and why pairs are adjacent

Two properties the ordering has to deliver, both verified when the file was generated rather than
argued for:

- **Every point's 5 runs land in 5 DISTINCT boots.** R1 step 5 asks for five independent runs over at
  least three boots. Rep-major ordering also means a wedged boot costs each point **at most one** of
  its five runs, instead of destroying one whole series.
- **No spatial/Sublet pair is split across a boot boundary** (0 of 45). The protocol requires the
  pairing be preserved; a pair measured on two different boots is a pair with a second variable in it.
  This is structural, not luck: a pair starts at line `18(r-1)+2k-1`, which is always odd and so can
  never be congruent to 0 mod 12.

Boot occupancy is `[12,12,12,12,12,12,12,6]` — boot 8 is the short one.

## THE GATE STRING MUST BE OVERRIDDEN

`board-r1e4.sh` defaults to `R1_QEMU_GATE="R1 lat"` and requires 5 matching lines in the emulator log.
**This harness emits no `R1 lat` lines at all** — its point records are `R1 s=...`. Left at the default
the driver refuses every boot with "the emulator run of this image is not on record". Set:

    R1_QEMU_GATE='R1 s=' R1_QEMU_GATE_MIN=5

`R1 lat` belongs to the latency ladder arm, not to the slots-and-pools release measurement.

## Invocation

    R1_IMG=<image> R1_HASH=1b7a04fe237e1580 \
    R1_LIST=$PWD/lists/r1-invocations.txt \
    R1_QEMU_LOG=<emulator boot.log> R1_QEMU_GATE='R1 s=' R1_QEMU_GATE_MIN=5 \
    R1_HOST=/tmp/capstone/q0/sqlite_host_rr.user \
    R1_BOOT=k  bash board-r1e4.sh

The image must be linked at `DOMAIN_BASE_VA=0x410000` — the default `0x10000` collides with the `k800`
control rung and preflight C15 refuses the boot.

## One control this bitstream cannot deliver

R1 step 1 pairs a legal point with an **invalid-access companion**. On `caplifive_m1_054cea69b.bit`
that companion **cannot trap**: R-35 is a defect in the revocation check on exactly this path
(`load_store_unit.sv:966-971`), so an access through a released reference is permitted. The companion
is measurable on the emulator, which enforces, and is **unmeasurable on this bitstream** — the same
standing as M1's condition 3, and for the same named reason. Report it as unmeasured, never as passed.
