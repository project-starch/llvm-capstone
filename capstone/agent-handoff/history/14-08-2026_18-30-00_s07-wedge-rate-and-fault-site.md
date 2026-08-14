# S-07: how often does it fire, and does it move?

Date: 2026-08-14. Bitstream `caplifive_12august.bit`. Branch `capstone-bootstrap`.
Binary under test: `G6.dom`, sha256 `f93a9188a9a4433c…`, **not rebuilt between boots** — verified
present in the initramfs by hashing the cpio members, not the overlay directory.

## Why measure a rate at all

S-07 is sporadic, so "SQLite's basic workload runs on silicon" is either a result or it is not,
depending entirely on a number nobody had measured. The prior record was six genuine executions
of this binary — one wedge, five passes — which is too few to distinguish 1-in-6 from 1-in-2.

The design, fixed before the first boot so it could not drift: each boot runs `L2.dom` as a
control and then `G6.dom` eight times. The driver stops at the first failure and a wedge takes
the core, so each boot yields a **censored run-length**, not eight samples. That is the intended
datum. A boot whose control fails is VOID. An R-16 entry stall (no `SQ: G/enter`) is excluded
from the denominator entirely — it is not a pass and not a wedge, it is an absence of a
measurement. Stop after three valid boots regardless of outcome.

## Boot 1 — control passed, one pass, then a wedge

Read from the driver's RUN-SCOPED transcript (`PROBE_SCOPED_OUT`), not the driver log:

| # | domain | verdict |
|---|---|---|
| 1 | `L2.dom` | RETURNED — control good, **boot VALID** |
| 2 | `G6.dom` | `obs=1517159939` = `0x5A6E0603` — three rows, rc=3 |
| 3 | `G6.dom` | **WEDGED** — entered (`SQ: G/enter`), never returned |

Latched trap state, read before release:

```
sw=255  TRAP LOG {seen,mcause[6:0]}  = 0x99   -> seen, mcause 25 (UNEXPECTED_OPERAND)
sw=196..203  trap mepc (LATCHED)     = 0x00000000839416a8
```

`mepc` decodes to a domain VA of **0x1516a8** (`mepc & 0x3FFFFF` + `0x10000`; the 4 MiB base
`0x83800000` is aligned, as an order-10 `__get_free_pages` allocation must be). Disassembling
`G6.dom` there:

```
  151690: cincoffsetimm  a1, s0, -0x40
  151694: ldc            a1, 0x0(a1)      <- the shared-region payload cap, reloaded from its slot
  151698: cincoffsetimm  a4, s0, -0x48
  15169c: ld             a2, 0x0(a4)
  1516a0: addi           a3, a2, 0x1
  1516a4: sd             a3, 0x0(a4)
  1516a8: cincoffset     a1, a1, a2       <== mcause 25: a1 is NOT_CAP
  1516ac: sb             a0, 0x0(a1)
```

That is `output_text+0xdc` — **byte-for-byte the instruction already recorded as S-07 instance 2**.

## What this changes

**The fault site is fixed, the trigger is not.** Both wedges ever observed from this binary land
at the same instruction. So S-07 is not "a tag is lost somewhere at random"; for a given image
there is a specific site, and what varies between runs is whether it fires. That is a much
stronger statement to hand the hardware side than the three-scattered-instances framing, and it
is consistent with it — the three instances came from three different builds.

**The wedging capability is the domain's own output writer, not SQLite.** `output_text` writes
result rows through the monitor-granted shared-region capability. It has nothing to do with
SQLite's internals, which is worth stating plainly: this binary's failure is in the reporting
path, not the database engine.

**Provenance is not the mechanism, though.** The obvious next thought — that a monitor-granted
cross-domain capability is the fragile kind — is already refuted by instance 3
(`sqlite3DbMallocRawNN+0xd8`, the lookaside `pSmallFree`), which is an ordinary heap-resident
capability. Provenance may modulate the rate; it cannot be the cause. This is written down
because it is the rung somebody will otherwise build next, and four such rungs have already come
back 65535.

## Boot 2 — four consecutive passes, then an entry stall

| # | domain | verdict |
|---|---|---|
| 1 | `L2.dom` | RETURNED — **boot VALID** |
| 2–5 | `G6.dom` ×4 | `obs=0x5A6E0603` each, byte-identical |
| 6 | `G6.dom` | **ENTRY STALL** (R-16) — no `SQ: G/enter`; NO VERDICT, excluded |

The latched state is the useful confirmation here, because "did not return" looks the same for a
wedge and an entry stall and the driver reads the debug mux for both:

```
sw=255  = 0x89  -> seen, mcause 9 = ECALL-from-S-mode
mepc            = 0xffffffff800072cc  (a KERNEL VA)
```

That is not a domain fault, it is ordinary kernel activity latched before the domain would have
run — exactly what an entry stall must look like, and exactly what the driver's own comment warns
will be misread as "the domain took a capability fault". Classification on `SQ: G/enter` and the
latched state agree, independently.

## Boot 3 — wedged on the first rep, at the same instruction, from a different physical address

| # | domain | verdict |
|---|---|---|
| 1 | `L2.dom` | RETURNED — **boot VALID** |
| 2 | `G6.dom` | **WEDGED** — entered, never returned |

```
sw=255 = 0x99  -> seen, mcause 25 (UNEXPECTED_OPERAND)
mepc           = 0x00000000835416a8
```

Boot 1 latched `0x839416a8`; boot 3 latched `0x835416a8`. The 4 MiB bases differ — `0x83800000`
against `0x83400000`, two independent `__get_free_pages` allocations — and **both decode to domain
VA `0x1516a8`**, `output_text+0xdc`.

Stated precisely, because the tempting overstatement is close by: this shows the site is a property
of the **image offset**, not of where the kernel happened to place the domain at 4 MiB granularity.
It does **not** exclude a cache-set-dependent mechanism — the low 22 bits of the two addresses are
identical, so every set index is the same in both. Anyone reaching for "the physical address is
irrelevant" from this data is reaching too far.

## Final result

| source | genuine | passed | wedged | entry stalls (excluded) |
|---|---|---|---|---|
| prior record | 6 | 5 | 1 | 1 |
| boot 1 | 2 | 1 | 1 | 0 |
| boot 2 | 4 | 4 | 0 | 1 |
| boot 3 | 1 | 0 | 1 | 0 |
| **total** | **13** | **10** | **3** | **2** |

**p(wedge) ≈ 3/13 ≈ 23% per execution of the basic workload.** All three boots had passing
controls, so none is void. The two entry stalls are excluded from both numerator and denominator;
counting them as failures would give 3/15 and would be wrong, since an image that never entered
says nothing about the code in it.

**Every one of the three wedges is at `output_text+0xdc`.** Not "mostly", not "two of three" —
all of them, across three separate boots and two distinct physical placements.

The earlier estimate of "roughly 5 of 6 succeed" (≈17%) was made on six samples; at 13 it is 23%.
Same order, and the qualitative claim — the basic workload usually completes and sometimes does
not — is unchanged.

Boots 2 and 3 were deliberately pure repetitions rather than a new rung: the rate is what the
claim depends on, and one boot is not a reason to abandon a design mid-experiment.

## An instrument note worth keeping

From boot 2 onward the console's connect-time replay contains a **previous boot with the same
`TEST n/9` shape**. Anchoring a regex in the driver log therefore fuses the replayed boot with
the live one and silently double-counts. Boot 1 survived this only because its replay was the
older 6-test boot. Classify from `PROBE_SCOPED_OUT` — that file exists for exactly this reason.

## The operand probe perturbs the workload — measured, not suspected (2026-08-14, later)

`cincoffset`'s guard has two arms (`capstone_flu_unit.anvil:29-31`: raise if rs1 is NOT_CAP **or**
rs2 is not-NOT_CAP), so every wedge measured so far is ambiguous between "the capability lost its
tag" and "the offset gained one". To settle it, `output_text` was instrumented to query the type of
both operands per iteration and, on a lost tag, reload through the same stack slot and query again.

**The instrument works.** Its control arm (`PS.dom`, `-DCAPSTONE_OPERAND_PROBE_SELFTEST`) returned
`0x5B400000` on two separate boots — the LCC field-1 query reports NOT_CAP on this silicon, the
counters accumulate, and the value survives the return path.

**The instrumented build does not reach the code under test.** One boot, control-validated,
alternating arms:

| arm | rows | SQLite | result |
|---|---|---|---|
| `L2` | — | — | returned — boot VALID |
| `P1` (probe) | 0 | `ERR create rc=11` | returned |
| `G6` (no probe) | **3** | clean | `obs=0x5A6E0603` |
| `P1` (probe) | 0 | `ERR create rc=11` | returned |
| `G6` (no probe) | **3** | clean | `obs=0x5A6E0603` |
| `P1` (probe) | 0 | — | entry stall |

`G6` succeeds twice in the boot where `P1` fails twice, so this is neither the board nor a toolchain
drift: **the ~85 instructions the probe adds to `output_text` are the whole difference**, and they
turn a working CREATE into `rc=11` (SQLITE_CORRUPT, malformed schema). Excluded without a boot: the
granule guard is present (512 `lcc` in `P1` against 511 in `G6`) and the R-14 workaround is
default-on.

### Why this matters beyond a failed experiment

1. **It is evidence about the defect.** The failure is **layout-sensitive**, not workload-sensitive:
   the same source, same flags, same guard, same board, differing only by instructions added to one
   function, fails at a completely different and much earlier point. That is the S-01
   image-perturbation class, and it means an S-07 mechanism proposal must explain sensitivity to
   image layout.
2. **It sets the protocol for every future in-frame instrument here.** An instrumented arm must be
   run **against its uninstrumented twin in the same boot**. Had `P1` been run alone — as the first
   probe boot did — `create rc=11` would have looked like a discovery about SQLite rather than an
   artifact of measuring it. The matched pair cost one boot and converted a false lead into a fact.

**Still open:** whether the untagged operand is rs1 or rs2. The instrument is proven; what is needed
is a way to ask that does not move the image — a strictly smaller probe, or a binary patch of the
`G6` bytes rather than a recompile.
