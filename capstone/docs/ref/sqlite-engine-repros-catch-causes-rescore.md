# The two SQLite silicon arms re-scored: a catch is the mechanism, not any fault

**For the silicon and board lanes.** A sibling of `fpga-silicon-measurements-for-paper.md` rather
than a section inside it: that file carries three pre-existing lines the repository's own
pre-commit hook refuses, so it cannot be appended to without bypassing a release gate. The content
is what would have gone in it, and `bug-corpora/sqlite/engine-repros/README.md` and
`docs/plans/sqlite-engine-repros-board-rerun.md` both point here.

## The two SQLite silicon arms are RE-SCORED, and nothing was re-measured: a catch is now the mechanism, not any fault (`silicon-sqlite-stock` −2 cells, `vp2-malloc` −5, 2026-10-31)

**For the silicon and board lanes: your two SQLite arms' published cells changed, and no board time
is owed.** Both bundles were re-folded, not re-run. `tools/derive-verdicts.py --refold` re-judges
each stored Observation and **refuses to alter one** -- a row whose observation would change aborts
the whole re-fold -- so what the boards measured is byte-identical before and after, and each
bundle's `inputs.json` carries a `refolds` entry naming every cell and why.

- **The rule.** `tools/arms.json` now gives `virtual-sqlite-memsys5`, `virtual-sqlite-memsys5-pools`
  and `silicon-sqlite-memsys5` a `catch_causes` of `[25, 28]`. Those are the only causes a Capstone
  mechanism reports: `25` INVALID_CAP is the lifetime check refusing the access, `28` CAP_OOB is the
  bounds check. `24` UNEXP_OP_TYPE is the architecture's baseline tag check firing on the
  CONSEQUENCE of an access that was already allowed, and on silicon the same event at the same pc is
  reported as `4` (load access fault) or `13` (page fault) -- Q-31. A fault with any of those three
  is a MISSED with the fault kept in the cell's `verdict_note`, not a catch.
- **`silicon-sqlite-stock`** (`results/2026-10-16-silicon/silicon-sqlite-stock`, configuration
  `silicon-sqlite-memsys5`): 2 cells, both `cause=24` -- 02 `backup_pager_after_detach` and 05
  `detach_with_pending_trigger`, CAUGHT -> MISSED. The bundle now tallies 8 MISSED and 0 CAUGHT.
- **`vp2-malloc`** (`results/2026-10-18-vp2-rdx0/vp2-malloc`, configuration
  `virtual-sqlite-memsys5`): 5 cells, CAUGHT -> MISSED -- 02, 05 and 24 `cause=24`, 19 `cause=4`,
  22 `cause=13`. The bundle now tallies 31 MISSED, 1 NO-READING:inconclusive, 1
  NO-READING:unattributed, and 0 CAUGHT.
- **The protected arm loses nothing**: `virtual-nested-pools` re-scored 0 cells, because every catch
  it recorded was already `25` or `28`. That difference is the whole distance between the base and
  Sublet columns, and it now lives in the cells rather than in prose.
- **Consequence for the base column.** The unprotected machine catches NOTHING on either axis of
  this corpus, on QEMU and on silicon alike: `virtual-malloc`, `vp2-malloc` and
  `silicon-sqlite-stock` all read 0 CAUGHT.
- **An earlier claim that this needed a board re-run is withdrawn.** The obstacle was a missing
  tool, not a missing measurement; a board bundle re-folds exactly like a QEMU one.
- Where it is written down: `bug-corpora/SCHEMA.md` ("Verdicts: one judge, one original") and
  `bug-corpora/sqlite/engine-repros/README.md`. Lane branch `sqlite-rerun-v1`.
