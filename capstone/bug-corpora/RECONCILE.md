# How the study and this tree's case set were reconciled

Written 2026-10-09. Kept because the reasoning is not recoverable from the
result, and because one part of it is still pending.

## The problem, and why the obvious reading of it was wrong

The three-arm study was built on a base that has since been merged, while
another lane grew and measured several of the same groups. A trial merge
reports seven `modify/delete` conflicts and reads as two irreconcilable case
sets: the study restructures twenty-two group declarations into nine, deleting
exactly the files the other lane had just written measurements into.

**Keyed on the BUG rather than on the case DIRECTORY, that reading is wrong.**
All 162 cases of the study exist here as the same bug -- same upstream id,
same slug -- and 0 are orphaned. 25 of the shared cases merely carry a
different leading ordinal, because deleting one case renumbers every directory
after it inside its group, and that renumbering is most of what the merge was
reporting.

`tools/reconcile.py` regenerates this against any tree in the older layout:

    tools/reconcile.py <other-tree>/capstone/bug-corpora --json reconcile.json

## How the other lane's arms map onto these three

This is what decides whether an open cell needs a run or only a reader.

| their arm | this study's arm | valid? |
|---|---|---|
| `sublet` | `capstone-sublet` | yes, direct -- its oracle names the Sublet system allocator and reports cause 24, an access through revoked authority |
| `cheribsd-revocation` | `cheribsd` | yes, direct |
| `spatial` | `capstone-sysalloc` | **only where nothing is freed** |

`spatial` is BOUNDS-ONLY: 80 of its 85 measured oracles say in so many words
that *free only marks the block free*, and `capstone-sysalloc` revokes
synchronously at free. On a spatial crossing the two cannot differ, so the
measurement reads across. On a temporal one their own text says *NOT CAUGHT; an
exact bound cannot see a dead object -- the stale capability still carries the
freed object's bounds, and level0 reissued the same block*, where an arm that
revokes would catch. **Reading it across there would understate the arm by up to
26 cases**, and the declarations say so per group rather than leaving it to be
rediscovered.

One consequence falls out immediately and is not yet applied: for all 26
temporal cases `cheribsd-revocation` reads *no fault, and no reuse -- the chunk
was never handed to another allocation, because CheriBSD's libc holds a released
chunk in QUARANTINE until the revoker has swept*. That is this study's
`quarantined-unswept` disposition, and under the rule of 2026-10-08 those cells
are **catches**. The disposition step is where that lands.

## Still pending: the 25 renumbered cases

The study deleted one `sqlite/engine-repros` case and every directory after it
shifted by one. Until the deletion step lands, the two numberings coexist, and
a declaration keyed on the directory rather than the slug reads the wrong rows --
which `capstone-sublet` on that group already did, reporting 5 catches where its
bundle holds 22. See OPEN.md.

| program/group | here | in the study |
|---|---|---|
| sqlite/engine-repros | `14_25e3073741_fts5_structure_release_refcount` | `13_25e3073741_fts5_structure_release_refcount` |
| sqlite/engine-repros | `15_2639ddc474_fts5_vocab_eof_stale_poslist` | `14_2639ddc474_fts5_vocab_eof_stale_poslist` |
| sqlite/engine-repros | `17_28001204f4_json_each_static_stale_key` | `16_28001204f4_json_each_static_stale_key` |
| sqlite/engine-repros | `25_415540ddaa_fts5_getvarint_overread` | `24_415540ddaa_fts5_getvarint_overread` |
| sqlite/engine-repros | `08_51dd67080a_fts3_offsets_stale_buffer` | `07_51dd67080a_fts3_offsets_stale_buffer` |
| sqlite/engine-repros | `26_8f5b14a5c2_fts5_decode_getvarint32_overread` | `25_8f5b14a5c2_fts5_decode_getvarint32_overread` |
| sqlite/engine-repros | `27_a783931794_fts3_getvarint_overread` | `26_a783931794_fts3_getvarint_overread` |
| sqlite/engine-repros | `10_becd68ba0d_fts3_snippet_or_stale_iter` | `09_becd68ba0d_fts3_snippet_or_stale_iter` |
| sqlite/engine-repros | `28_c7def600bd_fts3_matchinfo_overread` | `27_c7def600bd_fts3_matchinfo_overread` |
| sqlite/engine-repros | `19_c8c9cdd9dd_rtree_cursor_stale_node` | `18_c8c9cdd9dd_rtree_cursor_stale_node` |
| sqlite/engine-repros | `20_d21bd37c7c_rtree_inode0_stale_node` | `19_d21bd37c7c_rtree_inode0_stale_node` |
| sqlite/engine-repros | `24_d4b646997a_writable_schema_stale_table` | `23_d4b646997a_writable_schema_stale_table` |
| sqlite/engine-repros | `11_dee0359ddb_fts3_zterm_stale_buffer` | `10_dee0359ddb_fts3_zterm_stale_buffer` |
| sqlite/engine-repros | `23_eab0e10304_fts3_static_bind_stale_text` | `22_eab0e10304_fts3_static_bind_stale_text` |
| sqlite/engine-repros | `21_eab0e10304_rtree_static_bind_stale_text` | `20_eab0e10304_rtree_static_bind_stale_text` |
| sqlite/engine-repros | `12_fb8ca7de0c_fts5_inplace_stale_leaf` | `11_fb8ca7de0c_fts5_inplace_stale_leaf` |
| sqlite/engine-repros | `13_fix-2026-06-08_fts5_near_stale_phrase` | `12_fix-2026-06-08_fts5_near_stale_phrase` |
| sqlite/engine-repros | `22_fix-2026-07-26_spellfix_oom_stale_vtab` | `21_fix-2026-07-26_spellfix_oom_stale_vtab` |
| sqlite/engine-repros | `09_fix-2026-08-17_fts3_snippet_stale_buffer` | `08_fix-2026-08-17_fts3_snippet_stale_buffer` |
| sqlite/engine-repros | `29_fz02_btree_get4byte_page_overread` | `28_fz02_btree_get4byte_page_overread` |
| sqlite/engine-repros | `30_fz06_recordcompareint_overread` | `29_fz06_recordcompareint_overread` |
| sqlite/engine-repros | `31_fz10_dbstat_decodepage_overflow` | `30_fz10_dbstat_decodepage_overflow` |
| sqlite/engine-repros | `32_fz11_dbstat_freechain_overflow` | `31_fz11_dbstat_freechain_overflow` |
| sqlite/engine-repros | `18_mem5-design_memsys5_inband_freelist_overwrite` | `17_mem5-design_memsys5_inband_freelist_overwrite` |
| sqlite/engine-repros | `16_unfixed_json_each_root_stale_key` | `15_unfixed_json_each_root_stale_key` |
