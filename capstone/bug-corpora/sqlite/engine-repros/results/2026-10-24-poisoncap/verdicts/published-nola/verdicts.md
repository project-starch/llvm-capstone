| case | published | spatial | signed | attributable |
|---|---|---|---|---|
| 00_adfb203a7d_fts5_getvarint_stale_read | fault sqlite3Fts5GetVarint code=1 | not run |  |  |
| 01_5e4233a9e4_agginfo_stale_after_realloc (no qualified trigger) | returned | not run |  |  |
| 02_b9e0f62c3a_backup_pager_after_detach | fault sqlite3BtreeUpdateMeta code=2 | not run |  |  |
| 03_e464802d49_blob_close_after_db_close | fault sqlite3DbFreeNN code=3 | not run | yes |  |
| 04_8504d37b99_blob_write_after_reopen (no qualified trigger) | returned | not run |  |  |
| 05_6397a78b2b_detach_with_pending_trigger | fault findElementWithHash code=3 | not run | yes |  |
| 06_0c8c9a64b3_expert_rem_ctx_freed | fault idxRemFunc code=3 | not run | yes |  |
| 07_129371553c_fts3_destroy_oom_stale_table | returned | not run |  |  |
| 08_51dd67080a_fts3_offsets_stale_buffer (no qualified trigger) | returned | not run |  |  |
| 09_fix-2026-08-17_fts3_snippet_stale_buffer (no qualified trigger) | returned | not run |  |  |
| 10_becd68ba0d_fts3_snippet_or_stale_iter | fault fts3SnippetAdvance code=3 | not run | yes |  |
| 11_dee0359ddb_fts3_zterm_stale_buffer (no qualified trigger) | returned | not run |  |  |
| 12_fb8ca7de0c_fts5_inplace_stale_leaf | returned | not run |  |  |
| 13_fix-2026-06-08_fts5_near_stale_phrase (no qualified trigger) | returned | not run |  |  |
| 14_25e3073741_fts5_structure_release_refcount | fault fts5StructureRelease code=3 | not run | yes |  |
| 15_2639ddc474_fts5_vocab_eof_stale_poslist | returned | not run |  |  |
| 16_unfixed_json_each_root_stale_key | fault memcmp code=3 | not run | yes |  |
| 17_28001204f4_json_each_static_stale_key | fault memcmp code=3 | not run | yes |  |
| 18_mem5-design_memsys5_inband_freelist_overwrite | fault run_case code=3 | not run | yes |  |
| 19_c8c9cdd9dd_rtree_cursor_stale_node | fault sqlite3RowSetInit code=1 | not run |  |  |
| 20_d21bd37c7c_rtree_inode0_stale_node | fault nodeHashLookup code=3 | not run | yes |  |
| 21_eab0e10304_rtree_static_bind_stale_text | returned | not run |  |  |
| 22_fix-2026-07-26_spellfix_oom_stale_vtab | fault editDist3FindLang code=3 | not run | yes |  |
| 23_eab0e10304_fts3_static_bind_stale_text | returned | not run |  |  |
| 24_d4b646997a_writable_schema_stale_table | fault sqlite3OpenTable code=3 | not run | yes |  |

qualified 19: fault at site 13, mechanism-signed 11
