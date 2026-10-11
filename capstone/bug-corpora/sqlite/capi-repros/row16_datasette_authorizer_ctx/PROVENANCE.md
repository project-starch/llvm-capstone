# PROVENANCE — row16 (datasette-sqlite-authorizer)

- Cited upstream : https://github.com/datasette/datasette-sqlite-authorizer/issues/3
- Tier           : **MODEL (not traceable to a memory-safety PoC)**

**Honest label.** The cited issue #3 is a **functional test-failure** report
(read-only-protection tests failing on Python 3.11) — **no use-after-free
reproducer, no vulnerable source, no fix commit**. So row16's "authorizer context
UAF" is a CONSTRUCTED, mechanism-faithful MODEL of the SEALED-CALLBACK shape, not a
lowering of a documented CVE. Full rationale in the repro header
(`before-faithful.c`) and `design/sqlite-corpus-fidelity-ceiling.md`.
