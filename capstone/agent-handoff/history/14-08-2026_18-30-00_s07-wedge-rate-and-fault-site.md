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

## Running total

| source | genuine | passed | wedged | entry stalls (excluded) |
|---|---|---|---|---|
| prior record | 6 | 5 | 1 | 1 |
| boot 1 | 2 | 1 | 1 | 0 |
| boot 2 | 4 | 4 | 0 | 1 |
| **total so far** | **12** | **10** | **2** | **2** |

So p(wedge) per execution is **2/12 ≈ 17%**, and the README's "roughly 5 of 6 succeed" happens to
survive contact with four times the data. Boot 3 pending.

Boots 2 and 3 were deliberately pure repetitions rather than a new rung: the rate is what the
claim depends on, and one boot is not a reason to abandon a design mid-experiment.

## An instrument note worth keeping

From boot 2 onward the console's connect-time replay contains a **previous boot with the same
`TEST n/9` shape**. Anchoring a regex in the driver log therefore fuses the replayed boot with
the live one and silently double-counts. Boot 1 survived this only because its replay was the
older 6-test boot. Classify from `PROBE_SCOPED_OUT` — that file exists for exactly this reason.
