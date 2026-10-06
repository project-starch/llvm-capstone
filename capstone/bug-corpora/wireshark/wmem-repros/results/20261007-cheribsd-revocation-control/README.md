# All 18 wmem defects complete on stock CheriBSD — and this time with a revocation control

Stock CheriBSD purecap, **libc revocation ON**, `guest_default: preserved`, mode 0.
**21 of 21 arms passed, runner exit 0. 0 of 18 caught.**

| control | reading |
|---|---|
| `cheribsd-abi` | `CHERI_ABI pointer_bytes=16 runtime_revocation=1` — PASS |
| `cheribsd-bounds` | `CHERI_BOUNDARY_READY` — PASS |
| **`revocation-control`** | **FAULTED** — `tag_after_sweep=0 reissued=1`, then `SIGPROT` `si_code` 2 at `addr=pc=0x101dc0` |

## Why this run exists, when 13 of these cases were already measured

`results/20260921-cheribsd/` covers cases 0-12 at revocation **on**, and its reading is not in doubt.
What it does **not** contain is a revocation control. Its README cites a PoisonCap mode-0 *"matched
control"*, a mode-1 arm that faults, and *"the bounds control faulted in the same boot"*.

**A bounds control and a PoisonCap differential do not show that CheriBSD's revoker sweeps.** They
show that bounds are enforced and that the port's own protection behaves differently from its
baseline. Neither licenses the sentence a temporal row needs — *"the mechanism is active and did not
fire"* — because neither demonstrates the mechanism can fire at all. Without that, a completion is
indistinguishable from a revoker that was never doing anything.

That bar is the one the FFmpeg pool and memcached readings meet, and this run brings wmem to it. The
control mallocs a block, frees it, asks the revoker to sweep, re-allocates the same size and reads
through the **old** pointer: `tag_after_sweep=0` (the stale capability lost its tag) and then a fault
at an address equal to the one `supervise` resolved for the labelled probe independently of the run.

**Two further things this run fixes**, neither of which the 2026-09-21 bundle could:

- **The 5 spatial rows (13-17) had no bundle at all.** They were measured 2026-10-06 and recorded
  only in their `case.json`. All 18 are now in one record.
- **The vehicle now matches.** The 2026-09-21 bundle ran on image `3571a6d2…`, a different
  (PoisonCap-tree) image. This run uses `0cb16209…`, **byte-identical** to the vehicle the FFmpeg and
  memcached readings used — so the three programs' CheriBSD columns are comparable rather than merely
  all labelled "CheriBSD".

The earlier bundle is **kept**: it carries the PoisonCap arms, which this run does not measure.

## The reading, and why mode 0 is the right arm

Mode 0 is the port's **unprotected baseline** — objects are request-bounded and a packet-pool reset
revokes nothing — which is exactly the arm that asks what the *shipping* temporal mechanism does.
Every one of the 18 completes.

The mechanism is the same one memcached and FFmpeg's pools show: **the storage never reaches
`free()`**. A wmem scope reset returns chunks inside a block `malloc` still owns, so libc's quarantine
never holds the object and the revoker has nothing to sweep. Cases 13-17 are spatial and complete for
a different reason — a crossing inside a block the allocator still owns is in bounds for a
per-allocation capability.

**A caveat this corpus's own `corpus.json` already states, and which applies to this reading too:**
every arm of this harness narrows a wmem allocation to its request via `wm_narrow()`, so these rows
do not discriminate the chunk port. What they measure here is stock CheriBSD, which is the question
this bundle asks.

**N = 1 per cell.** The control is `ports/memcached/allocators/security-tests/cheribsd/revocation-control.c`,
borrowed unchanged — it exports its probe as a global asm label, so one supervisor resolves it
whichever corpus is under test, which the FFmpeg pool reading established.

Files: `result-lines.txt`, `inputs.json`.
