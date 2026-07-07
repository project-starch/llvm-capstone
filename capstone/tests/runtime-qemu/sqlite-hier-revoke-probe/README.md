# SQLite hierarchical-cascade feasibility probe (use-after-close)

A Stage-2 experiment for the HIERARCHICAL-REVOKE rows (use-after-close: cve-repros
rows 4/5/7/8/9/10/12). It asks whether revoking a **parent** (connection) borrow
cascades to a **child** (statement/value) borrow, using **only existing lender
ops** (no firmware change).

## Result (2026-07-07) — NO-CASCADE (independent rev roots)

```
connection (parent) borrowed to host
statement value (child) borrowed to host
round 1 retval = 0xc01a0dedc01a0ded          # child read while open
host read statement value OK before close
close revoked the connection (parent)         # revoke_region(parent)
round 2 returned 0xc01a0dedc01a0ded           # child STILL readable
NO-CASCADE parent revoke did not invalidate child (independent rev roots)
```

Two regions from separate `create_region` calls are **independent rev-tree
roots**, so `revoke_region(parent)` does not touch the child. A faithful
senior-cascade (the paper's H primitive: `close` invalidates every statement /
value beneath the connection) requires the child capability to be
**split-derived from the parent** so it is junior in the parent's rev lineage.
That needs a monitor extension (a "share child split from parent" op); see the
dated history note `agent-handoff/history/07-07-2026_*_hierarchical-cascade-*`.

This probe is retained as the negative-result characterization (like the M0
`borrow-revoke-uaf-probe`); when the derived-child op lands, add a positive probe
and flip the final marker to the TRAPPED diagnostic.

Build/run: `../build-sqlite-hier-revoke-probe.sh`, `../run-sqlite-hier-revoke-probe.sh`.
