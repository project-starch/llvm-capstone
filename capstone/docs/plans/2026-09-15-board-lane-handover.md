# Board-lane handover to `apollo-board` (2026-09-15, 15:45)

**For the successor session.** You take over the board lane: baking and booting the FPGA, reading runs,
bundling results in the paper's record shape, keeping the measurements doc, the registry and the state doc,
and coordinating with the other lanes. You are on the apollo server, so only the git repositories and the
board console reach you; everything you need is in the repo as of dev `49b40607817e` plus the commit that
adds this document, or is named below as something you create. **The board is one serialized resource: it
stays with the previous board session until your control-only boot (Q0) reads `done`; from then on it is
yours and that session is backup, hands-off.** Never two boots at once, across lanes.

## 1. Read first, in this order

1. `CLAUDE.md` — every hard constraint applies unchanged (no real names anywhere; the scan before every
   commit and push, by absolute path, gated on its exit status; never edit or push the paper; ask before
   editing CLAUDE.md; reflash ask-first; commit `-o` your own paths after reading `git diff`; no
   `Co-Authored-By`; background tasks end when their work ends).
2. `.claude/skills/board-run/SKILL.md` (bake → order → classify → release; the ordering rule; entry stall
   vs wedge) and `.claude/skills/rtl-sim/SKILL.md` (a directed test in 14 s; the three traps).
3. `docs/ref/HOW-TO-LAUNCH-ON-FPGA.md` — the full reference (NOT `agent-handoff/`, which no longer exists).
4. `docs/state/current-next-step.md` (its 15:40 and 14:45 headers) and `docs/state/current-state.md`.
5. `docs/plans/2026-09-15-sublet-paper-follow-on.md` — the plan you inherit: F1's gate (corrected
   14:15), F5's correction, F6's notes and pre-registration.
6. `docs/ref/fpga-silicon-measurements-for-paper.md` §7r–§7y — the last day's boots, and the shape of a
   §7 entry; `docs/ref/ISSUES.md` boxes M-11, C-32, Q-04, R-24, R-34 (and its "How to add an entry").
7. `tests/rtl-smoke/drivers/README.md` — the machinery, its knobs and gates.

## 2. Rules that lived only in the previous session's memory (one line each; the memory files are named)

* **Cite a board result by its image hash from the run-scoped transcript, never by a label**
  (`cite_results_by_image_hash_not_label`); a rebuilt image is a new hash and needs its own emulator pass
  (`stage_the_same_program_not_the_same_name`); pre-register decimals by machine (`preregister_decimals_by_machine`).
* **Per boot:** the rev-node table is 65,532 nodes with no reclamation and exhaustion is a deliberate stall
  (`rev_node_budget_65532`); ONE Sublet SQLite workload and ONE 128 MiB arena workload per boot; ≤ 12
  region-bearing harness invocations (M-9) and ≤ 80 % of the nodes cumulatively; distinct images need
  distinct entry VAs; a capability fault inside a domain WEDGES (M-1) — at most one expected-fault arm, last
  (`board_multi_stage_per_boot`, `batch_variants_one_board_session`).
* **Locks:** `sublet/r1/run-r1-qemu.sh` takes `$CAPSTONE_QEMU_LOCK` itself — never wrap it in an outer
  flock (a self-deadlock cost a morning; `qemu_lock_constant_path`); QEMU suites share one rootfs write
  lock, never two in parallel (`matrix_runs_serialize_rootfs_lock`); never rebuild the toolchain while a
  suite or a compiler-using run is in flight (`no_toolchain_rebuild_during_suite`); a bake waits on the
  machine memory lock (`MEMLOCK`) — on 2026-09-15 the lead directed the M2 bakes to a private lock once,
  which is the only exception on record.
* **Reading a run:** `driver.log` is read ONLY through `fpga_driver/transcript.py` — the console splits
  lines across chunks and the monitor's markers splice mid-token (M-11, `board_transcript_marker_splices`);
  `boot.txt` has no boot banner (a "banner count 0" there is structural); ugrep goes silent on control
  bytes — count with python, never `grep -c` (`rtl-sim` skill); watch UART-line growth for stalls, never
  file size (`detect_board_stalls`); an `ENTRY-STALL` verdict lives in `watchdog.log`.
* **Gates:** invoke `precommit-scan.sh` by absolute path and gate on its exit status (`scan && commit`,
  never a pipe); it reads REMOVED lines too, and the credential word followed by a colon or an equals sign BLOCKS in any text
  — reword (`precommit_scan_token_false_positive`, `gate_exit_status_and_removed_lines`); a
  worktree is scanned from INSIDE it; `grep -q` under `pipefail` gives a false BLOCK by SIGPIPE — use
  `grep -c` (`grep_q_pipefail_sigpipe_gate`); a gate override must not touch FAIL (`gate_override_must_not_touch_fail`).
* **Shell hygiene:** never `pgrep -f <literal>` in a tool call — it matches your own shell and kills it
  (exit 144, three times in one day); find PIDs by `pgrep -P`, pidfiles, or `ps -o cmd= -p` + `case`
  (`pgrep_pattern_anywhere_in_command`); never `kill -9` a board runner (SIGTERM by verified PID: `-9`
  orphans the server-side GDB session); never edit a running bash script (`never_edit_running_bash_script`);
  a Monitor or background task has an explicit end condition and exits itself — a polling loop with a
  file-based end test, not a `tail -F` pipeline (`monitor_tail_pipeline_lingers`).
* **Evidence:** a CLEAN result is not evidence until the check is known to fire (CLAUDE.md); ask what the
  instrument cannot distinguish BEFORE the claim goes out — M2's pre-registration mis-specified "touched
  nodes" and was corrected before the numbers were read (`one_arm_is_not_a_conclusion`,
  `read_the_producer_of_a_printed_figure`); "does not fire" and "raised and dropped" share an observable
  (R-34); a cause number in an expected-value field is a source-derived constant that needs the commit it
  was read at (`source_read_entries_go_stale`; cause 24 is this core's DEBUG_REQUEST); a simulation citation
  needs the artifact hashes at the moment of reading (`sim_artifacts_hash_at_read`); a surprising FAILURE
  needs a control as much as a surprising pass (`surprising_failure_needs_a_control`).
* **Repo hygiene:** other lanes may commit in the same checkout — a peer's commits sit under your own HEAD,
  so `HEAD..origin/dev` cannot show them; look at `<your last tip>..HEAD` (`session_split_compiler_vs_capstone`);
  `commit -o` scopes by path, not authorship — read `git diff <path>` before committing a shared file
  (`commit_only_scopes_by_path`); never `git checkout --` in a submodule (`never_checkout_submodule_source`);
  push only branches on `~/.claude-kisp/secrets/push-allowlist.txt`, which is the user's file — an agent
  never adds lines to it (`push_allowlist_only`); the paper submodule is never pushed; a tag publishes every
  commit it reaches (`tag_push_publishes_unscanned_commits`).
* **Writing:** results into the measurements doc need no permission; the paper's prose and `studies.json`
  are the paper lane's; a plan lists only the work; in a shared, concurrently appended doc never pre-assign
  a section number (`never_pre_assign_section_numbers`); one link per issue to the hardware side and the
  folder IS the report; a monitor change gets a claim-audit before it is baked; run the claim-auditor before
  a root cause enters ISSUES.md and name its weakest link.

The previous session's memory directory holds the full versions
(`~/.claude-kisp/projects/-home-<user>-dev-llvm-capstone/memory/` on the original host, 133 files); the lead may
copy it to your host, and you may read it there if you share the filesystem — you do not.

## 3. Where things live on your host (create what is missing)

* `~/.claude-kisp/secrets/{fpga-console-url,name-denylist.txt,push-allowlist.txt}` — you have them (report
  item 7). Never echo the URL; committed text uses `<FPGA-CONSOLE-URL>`.
* `${CAPSTONE_ARTIFACTS:-~/capstone-artifacts}/{unify,images,qemu-pass,bitstreams}/` — `unify/` holds one
  `board-<tag>/` per boot; `qemu-pass/<sha256>` is written by `run-r1-qemu.sh` on a passing emulator run and
  checked by every driver before a boot; `bitstreams/` you already hold and verified
  (`caplifive_r30r31_1bfff7776.bit` is the flashed one).
* The buildroot per-target layout (`sw/buildroot/build-fpga/…`, `overlay/test-domains/` — the drivers
  `mkdir -p` the overlay): your firmware is
  `build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin` and your module
  `build-fpga/target/capstone.ko` (report items 5–6); the drivers' defaults match, `CAPSTONE_BR_*` and
  `CAPSTONE_KO` override.
* `PREFLIGHT_ORACLES` (default `/tmp/capstone/ladder-fpga`): the k800 oracle and the `.qemu-pass` markers
  the preflight checks (C3/C4/C13). Regenerate them by the ladder's verify step under QEMU; never copy.
* The paper worktree: `git -C capstone/paper-nested-allocators worktree add ~/capstone-artifacts/paper-wt board/e1-s1s2-hardware`
  once the branch reaches you (§5); `make experiments-check` inside it before every bundle commit; scan from
  inside it. Note the submodule `capstone/paper-nested-allocators` lacks `update = none` and the paper repo
  refuses this account (403), so a fresh recursive clone aborts there — the lead holds the one-line fix.
* Toolchain bring-up on a fresh host: build `DEFAULT_TARGETS` **plus `llvm-config` and `llvm-readelf`**
  (without them `llvm-lit` runs zero tests and the stale-reader test fails; `DEFAULT_TARGETS` itself is not
  to be changed). Your build matches: harness hash `9a01b12a4db639b6` at `49b40607817e` (report item 3).
* Prerequisites still open on your host, the lead's to authorise: `pip install` of
  `tests/rtl-smoke/fpga_driver/requirements.txt` (python-socketio[client], aiohttp — without them no board
  connection), and a re-login for the `docker` group (RTL simulation; the `cva6-build-rv` image is staged).

## 4. Artifact lineage, by hash (cite these; a rebuild is a new hash)

| artifact | sha256/16 or commit | used for |
|---|---|---|
| R1 harness build7 / build8 / build9 | `3d6c24a64bceef00` / `55e6a187d52e5cc8` / `fdd3029ff0f96680` | E4 (§7v) / F4 (§7w) / M2's four boots (§7y) |
| R1 harness build10 / build11 | `848887ae81b8c0e3` / `9a01b12a4db639b6` | F6 flow checks; build11 = the timed M1 series (your rebuild matches it) |
| readback host `sqlite_host_rr.user` | `2c9e82d101b48160` | every R1/M2 boot; provenance: `sqlite_host.c` + the module's `libcapstone.c` at buildroot `d04bd83` |
| control rung `lpc` / `k800.dom` | `3b93a2b6e2adfa36` (PINNED: `tests/rtl-smoke/drivers/artifacts/lpc`, cannot be rebuilt) / (rebuilt; oracle `RESULT k800 retval=4`) | first and last in every boot |
| SQLite cells ⑤ -O2 / ⑥ -O2 / ⑥ -O1 / ⑥ E2 | `d61c8bf784f2bbd1` / `c506694f9f6f6889` / `902822a8f303dbaa` / `ceeded2533a74bce` | P1 (§7s) |
| C-32 pair (compiler lane) | before `ec061577fb008e18`, after `a1f8f2093696d511` | F1's gate reading (1/0 vs 1/1) |
| bitstream / monitor / FPGA buildroot copy | `caplifive_r30r31_1bfff7776` / `4274268` / `d04bd83` | the drivers check the last two by commit |
| board branch | `board/e1-s1s2-hardware` at `a9bafd8` | E1, R1, M2 bundles; the S1S2 and H1 manifests |

## 5. The queue (your plan), each with its gate

**Q0 — prove the environment, then one control-only boot.** Your eleven-line report stands at: PASS 1
(after the fast-forward), 2, 3, 4, 5–6 (with the layout knobs), 7, 8, 11-bitstreams; OPEN 9 (pip) and 10
(docker re-login), the lead's; 11-paper by bundle. Then, with the readback host rebuilt with `HOST_EXTRA_DEFS="-DSQLITE_HOST_REVOKE_RESHARE=1"` and verified
(`strings | grep -c 'RR/share'` ≥ 1; pass its hash by `R1_HOST_HASH`), `lpc` copied from `drivers/artifacts/` into the overlay,
the wrapper monitor copy at `4274268`, the ladder rebuilt and its oracle regenerated, the harness image's
emulator pass on record: `R1_BOOT=1 R1_OUT_TAG=q0 R1_IMG=<build11> R1_HASH=<its hash> R1_LIST=<an empty file>
R1_QEMU_LOG=<the flow check's drop boot.log> R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 R1_HOST=<rr host>
R1_BOOT_TAG=q0 R1_BOOT_DESC="apollo Q0: control only" R1_PREREG="retval=4 twice" bash tests/rtl-smoke/drivers/board-r1e4.sh`.
Pass = `RESULT k800 retval=4` twice, marker `done`, no runner left (check your task list). Tell the previous
board session and the lead; the state doc header then names you the board lane.

**Q1 — F1's confirming boot**, when the compiler lane says design A is on dev (blocked today on the lead's
ruling about collaborator author lines in the scan's range mode): rebuild ⑥ at pure -O2
(`ports/sqlite/build-sqlite-silicon.sh`, `SQLITE_OPT_LEVEL=-O2`, no `SQLITE_OPTNONE_FUNCS`), run
`tests/movc-cfg-scan.py` on it — the gate is **0 integer-only sites other than `renameResolveTrigger`'s
block-entry copy live around its back-edge**, enumerated by function and offset, with the mixed and opaque
buckets reported beside; the emulator pass is SQLLogicTest on the -O2 image with the `movc` density read
back (about 17,378 vs about 6,755 at -O0), not the standard suites; then ONE boot with `board-c6var.sh`
(control → ⑥-O2 at the 2 MiB arena → control → probe), pre-registered counters
5568/37966/32565/37966/5401 and cycles within 0.1 % of arm C's 1,376,190,813. Write §7s's reading, mark
C-32 FIXED with the scan as its gate, hand the paper lane the -O2 `protection_cost` at size 1.

**Q2 — the no-reclamation baseline**, ONLY on the lead's word (the paper lane has recommended it):
`tests/rtl-smoke/drivers/chain-m1.sh` with your build11 image and the flow check's log. Two boots; labelled
"no-reclamation baseline, NOT M1" everywhere; boot 1 diagnostic by capacity; primary = flatness of
`take_cyc/n + give_cyc/n` in cumulative allocations (slope × range < 1 % of the mean; last quarter within
1 % of the first), secondary = the sum in the band 384–391 raw cycles per allocation, scored apart; bundle
under `experiments/results/M1-baseline/fpga-<date>/` (`sublet/r1/m2-bundle.py` is the pattern for a
generator), a §7 entry, the state doc.

**Q3 — R-34's confirmation boot** after the RTL lane's fix is synthesised and the lead approves the reflash
(ask-first): the folder's test (`tests/fpga-repros/R34-lsu-exception-lost-on-immediate-grant/`) expects its
untagged arm to enter debug mode (R-24) and the others to trap; R-24 must be ruled before or with the fix.

**Routine, every boot:** the driver header carries the pre-registration; a §7 entry per boot (numbers in a
table, the reading in prose, what was NOT measured said explicitly); the state doc header; the registry box
if an issue moved; bundles on the board branch with `experiments-check` and the scan from inside the worktree;
push dev at stable points; a retraction is surfaced to the lead every time.

## 6. The lanes, and who owns what today (message by the `ListAgents` names)

* **paper** (`apollo-paper` as it comes live): the manuscript, `studies.json` evidence states, METHODS:86 with
  tab:safety, the M2 queried-set series (deferred behind M1), the S1S2 scope note; reads every §7 entry.
* **compiler**: C-32 design A (`46c53b7b6ae2`, `19bc05cf21b1` on their branch); tells you when it is on dev.
* **RTL** (`apollo-rtl`): R-34's fix sequenced with R-24; R-33's third arm (`2c59a355b`); the misaligned-
  under-translation test (done, 1e205738c667); their tree is `capstone-ariane` on `capstone-bootstrap`.
* **synth**: per-module area, the synthesis machine (leffe); never run synthesis from another lane.
* **helper**, **cheri**, **fpga**, **apollo-admin**: as the state doc names them.
* **The lead** (the user) holds the board; the questions listed in the state doc header are theirs.

## 7. The previous board session

Backup, hands-off after your Q0 passes: answers questions about the runs it made (§7r–§7y), the drivers and
the R-34 work; runs no boot. Its unpushed board branch is carried to you as a git bundle by the lead
(`~/capstone-artifacts/board-branch-2026-09-15.bundle` on the original host; `git bundle verify` then
`git fetch <bundle> board/e1-s1s2-hardware`).

## 8. The paper repository's operator procedure — read it before your first bundle (found 2026-09-15, 16:10)

The paper repository's remote is at **`7f83725` on BOTH `main` and `drafts`** (identical trees; an earlier
version of this section said `drafts` was 20 commits ahead — that compared a stale LOCAL `main`, and was wrong).
What is 20 commits behind is the platform repo's submodule gitlink for `capstone/paper-nested-allocators`
(`b0d7510c`) and the board branch's base. `7f83725` carries two files that no checkout of the board lane ever
contained and that the bundles of the last two days were made without: **`experiments/EXECUTION.md`** (the
operator instructions) and **`experiments/WORK-ORDER.md`** (the work-order template); read them from the paper
repository's remote (the bundle the lead carries to you holds `board/e1-s1s2-hardware` and `origin/drafts`,
which is that commit). **Landing the board branch is a MERGE, never a fast-forward:** relative to `7f83725` it
would delete `EXECUTION.md`, `WORK-ORDER.md`, `sections/`, `appendices/`, `macros/` and the rest of the
restructure that happened above its base. What the two files require, and what changes for you:

* **A work order per bundle, filled BEFORE measurement**, every pre-launch field marked REQUIRED resolved at launch:
  the study and protocol, the scope (audit / implementation / bounded measurement / full study), the question and
  the explicitly excluded claims, **operator AND reviewer** (the lead assigns both), the full paper commit, the
  platform checkout with every submodule commit, the manifest id and the new bundle directory, the board
  authorisation and resource budget; the inputs and gates (files and hashes, arm-to-binary mapping, sizes and
  layout, the complete `points.csv` and expected cell count, prerequisite results, positive controls and expected
  oracles, expected fault locations and survivor checks, counter and parser validation evidence, **peak and
  cumulative capacity calculation** — the node budget, stop conditions and recovery); the exact commands per stage
  with cwd and output destination; **timer start/end with exclusive nested brackets**, warm-up and deferred-work
  treatment, repetitions with seed and the complete boot order, the per-point timeout from the pilot, the plan if
  a control or capacity check fails; and the reviewer's acceptance at three gates before execution and one after.
  Q1, Q2 and Q3 above each get a work order first; ask the lead for the reviewer.
* **The parser is tested against a known-good log AND a deliberately wrong oracle** before a run (the transcript
  module's `test_transcript.py` is the known-good half; add the wrong-oracle half to your work order).
* **Never overwrite a previous bundle**; a superseding run is a new directory. **The FPGA launch gate names the
  actual launcher and its complete invocation** (`board-r1e4.sh` with every knob), and a failed run is stopped,
  released and preserved, then escalated to the reviewer with the study, point, source and image hashes.
* **Study ids are not platform issue ids**: `M1` is the reclamation study, `M-1` is the platform's trap-reporting
  issue, `sw74` names a boot; a record carries both `study_id` and `boot_id`. Both sets have been circulating in
  the same messages for two days — keep them apart in yours.
* **What the existing bundles lack against this procedure**, stated rather than repaired: E1, R1, H1 and M2 carry
  no `work-order.md` — they predate the procedure being visible to this lane, and a retrospective copy would be
  theatre, so they are not backfilled; no reviewer was ever assigned (the paper lane read everything, the role was
  not named); `S1S2/sw78-rep1` was removed when `sw78-rep1-3` superseded it (its records survive inside the
  successor; the rule is now known). The bundle path `experiments/results/<STUDY>/<RUN>/` and the manifest, runs,
  points, summary and raw layout match the procedure as written.
