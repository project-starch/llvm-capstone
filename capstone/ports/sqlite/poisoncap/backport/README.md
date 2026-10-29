# Backport demonstration: a real lookaside use-after-free on 3.22.0

This runs one case on both PoisonCap's published memsys5 and Sublet's memsys5 lookaside port, to show
what each does to a stale pointer that lives in a **lookaside slot** -- the per-connection pool of
small allocations. PoisonCap's `mem5.c` covers only memsys5 blocks, so it does not see a lookaside
slot reuse; Sublet's port issues and revokes every slot as its own child lifetime, so it does.

The case is a **backport**: a use-after-free that shipped in released SQLite 3.8.4-3.8.8, reverted
into the 3.22.0 engine by removing the one line its upstream fix added. Stock 3.22.0 carries the fix
and is not vulnerable, so this is never counted as a native 3.22.0 defect. The counted corpus stays
stock: the case lives here, not in `bug-corpora/sqlite/engine-repros/`, and the builders take it only when
`--extra-case` names its directory, so no corpus run includes it.

Proven natively only (2026-10-26). Neither platform run below has been made; their `expect` lines are
predictions, not results.

- Case: `33_f932f7145e_compound_cte_backport_pwith/` here (case.c, case.json, PROVENANCE.md); in
  `repro322/tags.tsv` it is the probe `pwithbackport`
- Patch (reverts upstream fix f932f7145e): `ports/sqlite/sqlite-pwith-backport.py`
- Prior art, constructed probes of the same layer (not counted): `repro322/case_lookaside_uaf.c`, `case_lookaside_tagmap.c`

## Native proofs (no platform needed)

```
# lookaside OFF + ASan -> heap-use-after-free in sqlite3WithDelete
# lookaside ON          -> the stale p->pWith lies inside [db->lookaside.pStart, pEnd), slot size 1200
```

## Build the backported amalgamation (per system, from the published amalgamation)

```
cp -r <published-amalgamation-dir> <backport-amalgamation-dir>
python3 ports/sqlite/sqlite-pwith-backport.py <backport-amalgamation-dir>/sqlite3.c
# asserts exactly one match; the two build tools then apply the usual FTS fixes on their own copy
```

## PoisonCap arm (p12, platform already built)

```
python3 ports/sqlite/poisoncap/corpus/build-cases.py \
    --platform ~/poisoncap-platform --amalgamation <backport-amalgamation-dir> \
    --ext-src <sqlite-3.22.0-src> --out <builds>/backport-pwith --only 33 \
    --extra-case ports/sqlite/poisoncap/backport/33_f932f7145e_compound_cte_backport_pwith
python3 ports/sqlite/poisoncap/corpus/run-cases.py <runs>/backport-pwith \
    --build <builds>/backport-pwith --expect published --cases 33
# predicted: controls gate passes; case 33 MISSED (returns, no poison report)
```

## Sublet arm (virtual paged platform)

```
python3 ports/sqlite/repro322/build-virtual.py --sublet \
    --sdk <virtual-sdk> --amalgamation <backport-amalgamation-dir> \
    --observe bug-corpora/sqlite/engine-repros/controls/control_uaf_mem5.c --observe bug-corpora/sqlite/engine-repros/controls/control_bounds_mem5.c \
    --out <builds>/backport-pwith-sublet --only 33 \
    --extra-case ports/sqlite/poisoncap/backport/33_f932f7145e_compound_cte_backport_pwith
python3 ports/sqlite/repro322/run-virtual-staged.py \
    --build <builds>/backport-pwith-sublet --platform <paged-platform> \
    --work <runs>/backport-pwith-sublet --llvm-bin <llvm-bin> --cases 33
# predicted: controls read their causes; case 33 CAUGHT (cause 25 at sqlite3WithDelete)
```
