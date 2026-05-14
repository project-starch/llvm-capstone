# History index and retention notes

This directory keeps timestamped session notes that may still be useful as primary sources.
It should **not** be part of the default startup reading set for a new chat.

## Default rule

Read `history/` only when one of these is needed:

- chronology,
- the original reasoning behind a decision,
- a previously validated command sequence,
- a root-cause trail that is not summarized in `current/`.

For normal startup, prefer:

1. `../README.md`
2. `../current/current-state.md`
3. `../current/current-next-step.md`

## High-value notes by topic

### Runtime blocker -> fix -> restored baseline

Read these if the session is about the old wrong-firmware / missing-Capstone-OpenSBI path:

- `12-05-2026_15-20-00_null_block_split_stock_opensbi_root_cause_and_runtime_status.md`
- `12-05-2026_16-05-00_local_mk_restored_opensbi_rebuild_and_null_block_split_success.md`
- `12-05-2026_17-05-00_post_fix_docs_cleanup_and_commit_plan.md`

### First validated HostCall proof sequence

Read these if the session is about the HostCall proof progression:

- `13-05-2026_14-10-47_first_hostcall_stdout_probe_validated.md`
- `13-05-2026_19-53-21_hostcall_stdout_borrowed_payload_validated.md`
- `14-05-2026_15-24-05_second_hostcall_service_filewrite_validated.md`
- `14-05-2026_16-47-11_reverse_direction_hostcall_fileread_validated.md`

### Earlier architectural exploration

Read only when reconstructing earlier design alternatives:

- `08-05-2026_14-31-30_split_roadmap_tradeoffs.md`
- `08-05-2026_15-13-28_split_rpc_roadmap.md`
- `08-05-2026_11-50-57_sbi_domain_abi_audit_and_return_probe.md`
- `08-05-2026_12-08-54_sbi_domain_shared_region_audit_and_annotated_probe.md`

### Legacy repro / diagnostic notes

These are mostly useful as archived supporting detail, not as default reading:

- `08-05-2026_11-21-40_hostcall_followup_and_blockers.md`
- `08-05-2026_11-21-40_shared_region_probe_result.md`
- `08-05-2026_12-19-23_null_block_reference_test.md`
- `08-05-2026_12-30-00_null_block_crash_reproduction_instructions.md`
- `11-05-2026_00-00-00_full_null_block_crash_reproduction_from_scratch.md`
- `12-05-2026_06-32-05_shared_region_and_null_block_diagnostic_snapshot.md`

## Retention guidance

- Keep `history/` as archival context, but do not duplicate its narrative into `current/`.
- If a history note becomes a near-duplicate of another note, prefer keeping one full primary source and replacing the duplicate with a short pointer.
- If a fresh session needs only the current state, do not read these notes eagerly.

