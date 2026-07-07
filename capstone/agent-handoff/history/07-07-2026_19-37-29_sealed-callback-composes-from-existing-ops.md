# SEALED-CALLBACK composes from existing ops (Stage-2, callback UAF)

**Date:** 2026-07-07
**Context:** SQLite-Capstone Stage-2. After BORROW-REVOKE (row 3) and
HIERARCHICAL-REVOKE (rows 4/5/7/8/9/10/12) were validated on RTL, the last
Stage-2 shape was SEALED-CALLBACK — the callback UAF rows 1/2/6/16 (cpython
progress-handler, rusqlite hook-closure, php UDF, datasette authorizer;
Table 4 primitive **S**, with L/R/H). This was the heaviest remaining unknown:
a callback is a bidirectional crossing, and the shape lists a sealed entry.

## Finding (empirical) — TRAPPED with existing ops

The feasibility probe `tests/runtime-qemu/sqlite-sealed-callback-revoke-probe`
(existing ops only, no firmware change) traps the callback UAF:

```
round 1 (invoke while registered) retval = 0xca11bac0ca11bac0   # callback reads pApp
callback unregistered (context revoked)                          # revoke_region == unregister
round 2 returned 0x000000000fa017ed                             # re-invocation TRAPPED
```

`round 2 == 0x0FA017ED` (fault sentinel) == the engine's sealed re-invocation of the
callback faulted on the revoked context; the monitor terminated the domain.

## Why it composes (no Step 2 needed)

SEALED-CALLBACK decomposes into pieces already validated:
- **Revocable context borrow** — the callback context (`pApp`) is lent
  `shared_region_annotated(PERM_IN, REV_BORROWED)` and ended with `revoke_region`
  at unregister/replace/close. This is exactly the BORROW-REVOKE mechanism
  (`sqlite-borrow-revoke-probe`), applied to the *context* rather than a row buffer.
- **Sealed invocation** — a domain is itself a sealed capability: the monitor's
  `create_dom` builds it with `__seal` (`capstone-sbi/sbi_capstone.c`), and every
  `call_dom` enters it via `__domcallsaves`. So invoking the callback is already a
  sealed-entry invocation; no new sealing op was required.

The planned Step 2 (a dedicated sealed-callback monitor op, on the pattern of
`share_child_region`) was therefore **not** needed — the probe trapped on the first
attempt with existing ops.

## Harness mapping

- `.user` lender = host binding: owns `pApp`, registers, invokes, unregisters.
- `.smode` sealed domain = engine callback body: caches `pApp` at registration and
  reads it on each sealed invocation.
- register == `shared_region_annotated(PERM_IN, REV_BORROWED)`; unregister / replace
  / close == `revoke_region`.

## Scope of the claim (stated to avoid overclaiming)

- **Buggy-host model (what Stage-1 rows 1/2/6/16 reproduce):** proven — the
  callback-context UAF becomes a deterministic fault on context revocation.
- **Malicious-host model:** the unforgeability of the callback entry and the
  inability to invoke a stale sealed one follow from the domain being a `__seal`-ed,
  provenance-carried capability — argued, not separately probed here.

## Status

All Stage-2 capability shapes are now validated on RTL:

| Shape | Rows | Status |
|---|---|---|
| BORROW-REVOKE (R) | 3, 13, 18, 19 | validated (`sqlite-borrow-revoke-probe`) |
| HIERARCHICAL-REVOKE (H) | 4, 5, 7, 8, 9, 10, 12 | validated (`sqlite-hier-child-revoke-probe`) |
| SEALED-CALLBACK (S) | 1, 2, 6, 16 | validated (`sqlite-sealed-callback-revoke-probe`, this note) |
| LINEAR (L) | 11 | not started (double-free via linear exclusivity) |
| UNINIT (U) | 14 | not started (use-before-init via uninitialised cap) |
| N/A | 15, 17 | out of scope (liveness / non-memory-safety) |

Remaining: the two small self-contained shapes LINEAR (row 11) and UNINIT (row 14),
neither of which needs the revoke scaffold. The three revocation-family shapes that
carried the paper's load-bearing risk are all retired.
