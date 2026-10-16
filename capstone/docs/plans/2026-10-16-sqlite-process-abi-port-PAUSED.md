# SQLite process-ABI port (wave 5 item 1) — PAUSED by the lead, 2026-10-16

Paused with every other wave-5 whole-program port: silicon is to run the full virtual Capstone
profile first, after which the virtual images run as built and these ports are not needed.
**This note lives on the branch, not on dev** — nothing half-finished was landed.

## Done, and already on dev

**`423d13f5b81f` — the sorter 16-alignment fix in `ports/sqlite/adapt-sqlite-322.sh`.** Landed on its
own because it is a correctness fix, not part of the port: the sorter advanced its memory cursor by
`ROUND8(nReq)`, so a record holding a capability landed 8-aligned and the ldc/stc on it is an
unaligned capability access. It also makes the virtual arms' input reproducible:

    stock 3.22.0 sqlite3.c                7991acd0d8c78435...
    + the script BEFORE 423d13f5          b446615b077da476...
    + the script AT 423d13f5              7fc92e19a3af61f3961b83447a91f1e1dec06646e343037dfd96a397167700b2

the last matching the `amalgamation_sha256` of nine published virtual bundles on all 64 digits.
The added assertion is negative-tested: against a source whose sorter line cannot match it fires,
the script exits 1, and no half-written `sqlite3.c` is left behind.

## Established, needs no repeating

- **`SQLITE_SUBLET_PATCH` takes the memsys5 patch.** `build-sqlite-silicon.sh:86-87` is a generic
  hook (`patch -s -F0 -p1`) into the build's PRIVATE copy of the amalgamation.
- **The 3.22.0 silicon path already uses that script.**
  `tests/rtl-smoke/repro322-silicon-2026-10-01/build-images.sh:24` exports
  `SQLITE_VERSION=3220000 SQLITE_YEAR=2018` and calls it per case, gp-captable and one TU. It simply
  never passes the patch — which is why that arm is the unprotected control. **The revoking arm is a
  parameter on an exercised path, not a new build.**
- **`sublet-3220000-memsys5.patch` applies 19/19 hunks** to 3.22.0, with AND without the two python
  pre-patches (`fts5-azarg`, `fts3-doclist-null`) — so those are NOT a prerequisite, despite the
  header's `Applies-To` reading like one. Control: the same dry-run against 3.53.3 **refuses**,
  12/19 hunks failed, so `-F0` is protective in practice and the clean result is real.

## Left to do, if this resumes

1. **Relabel the patch header.** `sublet-3220000-memsys5.patch` line 3 says
   `Upstream-SHA256: 7fc92e19... (sqlite3.c)`. That hash is the ADAPTED `--amalgamation` input
   (stock + `adapt-sqlite-322.sh`), not stock 3.22.0's `sqlite3.c` (`7991acd0`). Rename what it
   points at; the hash itself is now correct and reproducible.
2. **Build the revoking arm:** `SQLITE_SUBLET_PATCH=ports/sqlite/sublet/sublet-3220000-memsys5.patch`
   on the repro322 silicon path, input hash recorded in the build manifest. If the input hashes to
   `b446615b` the checkout is pre-423d13f5 — stop.
3. **The port proper:** extra TUs (cases 06, 22), FTS stubs (12), float for rtree (19-21), the B0
   pipeline (gp-captable, full LTO, soft float, check-repr full+renorm, no-fp-gate), and a qemu-pass
   per feature group on `capstone-qemu silicon/cderive-round ce1be0c642` with
   `CAPSTONE_CDERIVE_ROUND=1` and `GP_FABRICATE=0`.

Planning estimate was 1-2 days, from reading only; no measured figure exists because no qemu-pass
was reached before the pause.
