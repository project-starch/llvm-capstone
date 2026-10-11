# memcached's eight allocator defects on stock CheriBSD: 0 of 8 caught, with both controls

Stock CheriBSD purecap, **libc revocation ON**, `guest_default: preserved`, mode 0. All eight cases
complete. **0 of 8 caught.**

| control | reading |
|---|---|
| `cheribsd-abi` | PASS |
| `cheribsd-bounds` | PASS |
| `revocation-control` | **FAULTED** — `exit=162 signal=34 code=2 addr=0x101e42 pc=0x101e42 expect=0x101e42` |
| `negative-control` | **all 8 case oracles FAIL**, while the three above still PASS |

**Both halves of the two-sided argument are present, which is what this bundle adds.** The
revocation control proves the revoker *sweeps* in this guest — and `addr == pc == expect`, the
expectation having been resolved from the ELF independently of the run, so the fault is attributed.
The negative control proves the eight completions are *not vacuous*: with a fixture whose header
claims two events while the file carries one, the program refuses the input before any cache is
created (`runners/cheribsd/run-defects.py:80-88`) and **every** case oracle fails. So a completion
here means "the mechanism is active, it can fail, and it did not fire" rather than "nothing was
tested".

The mechanism: the storage never reaches `free()`. `cache.c` pushes an object onto its own STAILQ and
pops it back uncleared; `slabs.c` pushes a chunk onto its class's slots list and pops it straight
back. libc's quarantine never holds the object, so the revoker has nothing to sweep. The per-case
reuse counters are what make that concrete rather than asserted — `object_reuses=1` on cases 0 and 1,
`chunk_reuses=1` on cases 2, 3 and 4: the storage really was handed back out, which is exactly the
situation revocation would have to catch.

## Why this bundle did not exist until now

This corpus's `.gitignore` ignored `results/` outright — **the only one of sixteen corpora to do
so**, and the reason `docs/ref/cheribsd-denominator-audit.md:22` recorded memcached as *"declared and
claimed, no committed bundle"*. The ignore arrived with the initial port commit `2612d2d0318d`, whose
message gives no reason for it, and this corpus's own sibling `memcached/plain-heap-repros` commits a
bundle. So it was an inconsistency rather than a policy, and it is now narrowed to the raw guest
output and fixtures — which is what CLAUDE.md asks for: *"commit result lines, not the capture they
came from."* The narrowing was tested two-sided: `results/*/guest/` and `results/*/fixtures/` are
still ignored, `result-lines.txt` and `inputs.json` are not.

The readings themselves were taken on 2026-10-06 and have been in the case files since; what was
missing was the evidence beside them and the negative control, which was run for this bundle.

## Platform

Image `0cb16209c16c5edf9f81148eedf33f750af404f890af6d65d9bd43ecf61aa4f4`, **byte-identical** to the
vehicle the FFmpeg readings of the same day used, so the two programs' CheriBSD columns are directly
comparable rather than merely both labelled "CheriBSD". Kernel `CHERI-PURECAP-QEMU` from
`releng/26.07-88f39900c329`, which is `pins.env`'s `CHERIBSD_REV`.

**N = 1 per cell.** A revocation-OFF arm exists for the same eight cases as a companion differential
and is not required for this reading.

Files: `result-lines.txt`, `inputs.json`.
