# PostgreSQL's c-repros on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 61f0433d704c, before the boot). One boot, 16:33-16:41:
- firmware 1d2bbbf37f63f435, Image 0d9e05403ad6aeba;
- two arms, 28 runs, every one ended by itself (each arm's rung returned 0).

**Every row and every control is as pre-registered,** by outcome, cause and site, on the pre-registered image
(sha256): 5 cases and 5 negative controls per arm, plus the four tool controls of each.

| case | pgc-silicon-level0 | pgc-silicon-lifetimes | its negative control |
|---|---|---|---|
| 00 pqfn unbounded result copy | CAUGHT 28 in memcpy (declared) | CAUGHT 28 in memcpy (declared) | completes on both |
| 01 pg_dump transform list overread | CAUGHT 28 in pgclient_case_run | CAUGHT 28 in pgclient_case_run | completes on both |
| 02 ecpg bytea negative length | CAUGHT 28 in ecpg_hex_decode | CAUGHT 28 in ecpg_hex_decode | completes on both |
| 03 saslprep truncated UTF-8 overread | CAUGHT 28 in pg_utf8_islegal (declared) | CAUGHT 28 in pg_utf8_islegal (declared) | completes on both |
| 04 bbstreamer tar trailer stale pointer | CAUGHT 28 in pgclient_consume_content | CAUGHT 28 in pgclient_consume_content | completes on both |

Every unmarked site is the function the case names before its access (`expect_fault_in`).

**What it shows:**
- PostgreSQL's real libpq, pg_dump, ecpg, saslprep and tar-streamer functions run on silicon in a capstone-exec
  domain, and per-object bounds catch all five defects at the function their reduction names.
- Each case's own negative control, with the one value moved to the safe side of the boundary, completes. So each
  fault is the defect's, not the setup's.
- Case 04's stale pointer is caught by bounds (28) on both arms, as its recorded `spatial` arm was: the stale read
  leaves the old object's extent.

**Judged by** `tools/run-silicon-cases.py board --prereg` on the run-scoped transcript, then compared row by row with
the pre-registered outcome, cause and site, keyed by image hash.
