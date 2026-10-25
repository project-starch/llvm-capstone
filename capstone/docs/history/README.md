# History index and retention notes

This directory keeps timestamped session notes that may still be useful as primary sources.
It should **not** be part of the default startup reading set for a new chat.

## Default rule

Read `history/` only when one of these is needed:

- chronology,
- the original reasoning behind a decision,
- a previously validated command sequence,
- a root-cause trail that is not summarized in `state/` or `design/`.

For normal startup, prefer:

1. `../README.md`
2. `../state/current-state.md`
3. `../state/current-next-step.md`

## High-value notes by topic

### Runtime blocker → fix → restored baseline

Read if the session is about the old wrong-firmware / missing-Capstone-OpenSBI path:

- `12-05-2026_15-20-00_null_block_split_stock_opensbi_root_cause_and_runtime_status.md`
- `12-05-2026_16-05-00_local_mk_restored_opensbi_rebuild_and_null_block_split_success.md`

### First validated HostCall proof sequence

Read if the session is about the HostCall proof progression:

- `13-05-2026_14-10-47_first_hostcall_stdout_probe_validated.md`
- `13-05-2026_19-53-21_hostcall_stdout_borrowed_payload_validated.md`
- `14-05-2026_15-24-05_second_hostcall_service_filewrite_validated.md`
- `14-05-2026_16-47-11_reverse_direction_hostcall_fileread_validated.md`
- `14-05-2026_19-10-00_hostcall_metadata_snapshot_hardening_and_multi_pending_limit.md`
- `14-05-2026_20-05-00_second_pending_and_payload_reuse_diagnostics.md`

### CoreMark PureCap bring-up

Read if the session is about the backend compiler workarounds or CRC validation:

- `05-06-2026_00-00-00_coremark_purecap_bringup_complete.md` — CoreMark milestone:
  platform differences, 13-row workaround catalog, validated CRCs, commits.
  For the distilled design decisions see `../design/coremark-purecap.md`.

### Archived backend status memo

- `05-06-2026_01-00-00_capstone_backend_status_archived.md` — Backend/compiler status memo
  archived from `current/`. Written at `8b7450001ade`; Section 12 contains stale "verified"
  claims (frame lowering). See `../plans/backend-compiler-fixes.md` for current tracking.

### Earlier architectural exploration

Read only when reconstructing earlier design alternatives:

- `08-05-2026_14-31-30_split_roadmap_tradeoffs.md`
- `08-05-2026_15-13-28_split_rpc_roadmap.md`

### null_blk repro notes

Useful if null_blk split breaks again and you need the crash reproduction sequence:

- `08-05-2026_12-19-23_null_block_reference_test.md`
- `08-05-2026_12-30-00_null_block_crash_reproduction_instructions.md`

### The state and next-step journals before 2026-10-01

The two startup files were append-only journals and had grown to 2,030 and
1,560 lines, most of it superseded. On 2026-10-13 everything in them dated
before 2026-10-01 moved here verbatim, each with an index of its own sections:

- `13-10-2026_01-00-00_state-journal-through-2026-09-29.md` -- 58 sections.
- `13-10-2026_01-00-00_next-step-journal-through-2026-09-29.md` -- 20 sections,
  five of them headed `0. CURRENT`. **Do not take a next step from it**: a
  superseded entry in a next-step file is an instruction, and the board it
  instructs has been reflashed since.

The cut was by date alone, so neither file's contents were judged: a section
that stayed is not thereby current, and one that moved is not thereby wrong.

## Retention guidance

- Keep `history/` as archival context, but do not duplicate its narrative into `state/` or `design/`.
- If a history note becomes a near-duplicate of another note, prefer keeping one full primary
  source and reducing the duplicate to a short pointer.
- If a fresh session needs only the current state, do not read these notes eagerly.
