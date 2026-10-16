# PostgreSQL's c-repros on silicon: pre-registration (2026-10-16, before the boot)

The board campaign's wave 4 (`docs/plans/2026-10-16-board-campaign-all.md`, §3). Five memory-safety defects in
PostgreSQL 17.5's frontend and shared code (libpq, pg_dump, ecpg, saslprep, the basebackup tar streamer). Each
`case.c` drives the real upstream function, compiled from the pin (`shared/upstream_*.c`), and the defective object
comes from malloc.

Two arms, one boot on `caplifive_sublet_d93a8d66b`: private image 0d9e05403ad6aeba, firmware 1d2bbbf37f63f435, on
the exact-regions supervised-call line of sets 2-6.

| arm | heap | predicted from |
|---|---|---|
| pgc-silicon-level0 | level0, per-object bounds | the case's `spatial` arm (app-level0) |
| pgc-silicon-lifetimes | level0 + per-object lifetimes (set 4's placement) | the case's `virtual-malloc` arm |

**The protocol** (`tools/run-silicon-cases.py`, kind `pg-c`):
- **The build:** each case is `case.c` + `shared/driver.c` + every `shared/upstream_*.c` at -O0, as
  `shared/build-domain.sh` builds it.
- **The case run:** `<image> N`. It prints `PG_DEFECT case=N expect_fault_in=<function>` before its access, and
  `case N RETURNED` if nothing faulted.
- **Attribution:** a fault counts in that function, or in a `fault_sites` entry of case.json: 00 `pqGetnchar` /
  `memcpy`, 03 `pg_utf8_string_len` / `pg_utf8_islegal`.
- **The negative control:** each case is also built with `-DPGCLIENT_NEGATIVE_CONTROL`, which moves one value to the
  safe side of the boundary. That image must complete. It runs every time, not only on an unattributed fault, and
  gates the arm.
- **The tool's four controls** run on each arm as in sets 1-6.

## The silicon-mimic QEMU reading (capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`)

On both arms, all five cases are CAUGHT where the case says, every negative control completes, and the tool controls
are as declared:

| case | where | QEMU cause |
|---|---|---|
| 00 pqfn unbounded result copy | `memcpy` (declared) | 7 |
| 01 pg_dump transform list overread | `pgclient_case_run` (expect_fault_in) | 5 |
| 02 ecpg bytea negative length | `ecpg_hex_decode` (expect_fault_in) | 5 |
| 03 saslprep truncated UTF-8 overread | `pg_utf8_islegal` (declared) | 5 |
| 04 bbstreamer tar trailer stale pointer | `pgclient_consume_content` (expect_fault_in) | 5 |

Case 04 is a stale pointer, and both arms catch it by bounds (5), as its recorded `spatial` arm did. Its stale read
leaves the old object's extent.

## The board prediction

Every row as QEMU read it, by outcome and site, with 5 / 7 read as 28. No 24 occurs in a case row. The uaf control on
the lifetimes arm reads 25.

**What a failure would say:**
- **A negative control that faults** means the case's fault is not its defect, so the row cannot count.
- **A case that RETURNS** means the access stayed inside the bounds silicon gave the object.
