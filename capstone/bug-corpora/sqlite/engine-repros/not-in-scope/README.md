# Outside this study's scope

This study covers spatial and temporal memory-safety defects. These five are
neither: there is no heap object whose allocator boundary the access could be
inside or outside of, so the nested/non-nested question does not apply to them
and they cannot appear in any cell of the four-bucket table.

| case | shape | why it is not in scope |
|---|---|---|
| `40_174c21ff06_fts3_corrupt_image_hang` | non-terminating loop | no bad access at all |
| `41_fz12_fts5vocab_self_reference_recursion` | unbounded recursion | the exhausted resource is the C stack, not the heap |
| `42_fz13_dbstat_nonterminating_walk` | non-terminating loop | no bad access at all |
| `36_fz08_cursormoveto_wild_pointer` | wild pointer | dereferences the literal address `0x3`; neither out of bounds of an object nor a use after its lifetime |
| `37_fz09_searchwith_null_deref` | NULL dereference | a validity bug, not a bounds or a lifetime one |

The last two were not obvious from the `shape` field, which calls both "stale
or wild pointer dereference" together with two genuine out-of-bounds reads
(`27_33cf194218`, `29_634ac14488`). They were separated by reading each title.

They are kept rather than deleted. All five carry measured results on three
arms, they remain real defects in SQLite 3.22.0, and `41` and `42` in
particular are the only hangs in the collection -- useful for saying what the
mechanisms do with a program that never returns. Nothing reads this directory:
the checker discovers corpora by `corpus.json` and there is none here.

The same line was drawn for PostgreSQL in `postgres/sql-repros`, where the
reviewer asked for null dereferences, integer overflows, type confusions and
authorization defects to be removed for the same reason. Applying it to SQLite
too keeps one rule across the study instead of one per program.
