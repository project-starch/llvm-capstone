# FFmpeg's first CheriBSD readings: 3 of 4 caught, 1 refuted, and attribution established

Stock CheriBSD purecap, **libc revocation ON**, `guest_default: preserved`. The first CheriBSD
measurement of any FFmpeg corpus — until this run FFmpeg contributed **zero** measured rows, which is
what the 2026-10-06 retraction of `pool-repros`' claimed arm established.

| case | upstream | crossing | CheriBSD | attributed? |
|---:|---|---|---|---|
| 0 | `d133b4a231` | 4 B past a 16 B request | **CAUGHT** — `SIGPROT`, `si_code` 1, `addr`=`pc`=`0x1020ba` | **yes** |
| 1 | `bcbf3a5630` | 1 B past a 24 B request | **not caught** — completes, `exit status=0` | probe resolved, never faulted |
| 2 | `56309e476a` | 4 B **below** the base | **CAUGHT** — `SIGPROT`, `si_code` 1, `addr`=`pc`=`0x1020de` | **yes** |
| 3 | `495b402f27` | 33 B past a 512 B request | **CAUGHT** — `SIGPROT`, `si_code` 1, `addr`=`pc`=`0x102120` | **yes** |

`si_code` **1** is the BOUNDS code, matching `wireshark/plain-heap-repros/00`. Every runner in this
tree still hardcodes `PROT_CHERI_TAG = 2`, the *tag* code.

## Attribution is established here, and it was not in the sibling corpora

In all three catches the fault address **equals** the probe address `supervise` resolved from the ELF
*independently of the run* — `expect ffh_read_probe 0x1020ba` against `addr=pc=0x1020ba`. SCHEMA
rule 2, the fault at the labelled probe and nowhere else, is met.

The reason is a deliberate difference from `memcached/plain-heap-repros` and
`wireshark/plain-heap-repros`, whose rows read *"attribution: not established"*: those declare their
probes `static` in the header, so every translation unit gets a private copy and the symbol cannot be
resolved unambiguously. This corpus **declares** the probes in `shared/corpus.h` and **defines them
once** in `shared/driver.c`. The native readings were re-run after that change and came out
byte-identical, so the linkage change is inert to everything except attribution.

## The refutation, and why it is clean rather than embarrassing

**Case 1's committed prediction was CATCH, and it is wrong.** The mechanism was measured in the same
boot rather than inferred — CheriBSD's `malloc` bounds a capability to the allocator's **usable
size**, not to the request:

| request | capability length | slack |
|---:|---:|---:|
| 16 | 16 | **0** |
| **24** | **32** | **8** |
| 512 | 512 | 0 |
| 2048 | 2048 | 0 |

Case 1 requests 24 bytes and reads offset 24. The capability is 32 long, so offset 24 is **inside the
bounds and no fault is possible**. Cases 0 and 3 request sizes that land exactly on their class, so
their crossings leave; case 2 crosses *below the base*, which no size class can extend.

**The refutation was pre-registered.** The prediction committed this morning said of this row that 24
bytes "is not a size-class boundary, so the usable size must be read in-guest before this is
believed". It was read, and it was 32. **The calloc table measured for the memcached and wireshark
rows was deliberately not reused**, for exactly this reason: a size class is a step function and
those readings were taken at 1, 9, 16, 17 and 8192.

This is the **second** instance of this mechanism — `memcached/plain-heap-repros/00` was refuted by it
at request 9 → length 16 — so it is now measured at two different size classes and is a property of
the allocator's policy, not a one-off.

## The controls fired, and a first attempt was discarded because they did not

| control | reading |
|---|---|
| `cheribsd-abi` | `CHERI_ABI pointer_bytes=16 runtime_revocation=1` — **PASS** |
| `cheribsd-bounds` | `CHERI_BOUNDARY_READY` — **PASS** |

**A first run of this suite had both controls FAIL** and is not reported as data. The cause was mine:
the corpus's own size probe had been passed as `--abi-probe`, so the runner asked it for a marker it
does not print. Its case readings happened to be identical to the valid run's, which is exactly why
discarding them mattered — a result whose platform controls did not fire is not a reading, however
plausible it looks.

## What this does and does not establish

- **Does:** 3 of 4 caught with the fault attributed to the labelled probe; 1 not caught, with the
  mechanism measured at its own request size.
- **Does:** that "spatial is a tie with CHERI" holds only for crossings that leave the **usable**
  allocation — now shown twice, at two size classes, and shown not to apply below the base.
- **Does not** measure Capstone, PoisonCap, or the other 11 FFmpeg spatial rows. PoisonCap remains
  unavailable on this host; `subobject-repros` needs the port's purecap build, which is untested.
- **N = 1 per cell.**

Files: `result-lines.txt` (result lines only — the serial capture is not committed, it carries host
and account banners), `inputs.json`.
