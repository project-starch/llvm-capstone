# Plan: hand the board lane over to `apollo-board` (2026-09-15, afternoon)

## Context

The lead is moving the lanes to the apollo server; `apollo-board` (a Remote Control session on that
machine, visible to `ListAgents`) takes over this lane's work and must be able to do everything this
lane does: bake and boot the FPGA, read runs through the transcript module, bundle results in the
paper's record shape, keep the measurements doc, the registry and the state doc, and coordinate with
the other lanes. For a transition period every lane stays live (paper, compiler, RTL, synth, helper,
cheri, and the apollo-* set); the switch is gradual. This session stays as backup with the board
hands-off after the handover lands.

Two things make this more than "send the state doc": the successor is on ANOTHER MACHINE, so only the
git repositories and the board console reach it — and the lane's execution machinery has never been in
the repo. The board drivers live in `~/capstone-artifacts/unify/` (108 scripts; nine are the live set),
the chain scripts, invocation lists and the prepared M1 driver live in this session's `/tmp` scratchpad,
the readback host binary the drivers hash-check sits in that scratchpad too, the preflight's oracle
records are under `/tmp/capstone/ladder-fpga`, and every driver hard-codes `/home/<user>/…` and this
session's scratchpad path. Committed as they are they would be blocked by the name scan (the home
directory name is on the denylist) and would not run anywhere else. So the handover is: make the
machinery portable and commit it, write one handover document that carries what the docs and memory
do not, verify the successor's environment with positive controls before it spends a boot, and hand
the board over explicitly (one board, serialized: it is mine until the successor's control boot passes).

What the successor inherits, as of dev `49b40607817e` and the board branch at `a9bafd8`:

* **Done today:** F2 (M-11 structural), F3 (bundle hygiene), F4 (R-21/R-22 on silicon, §7w), F5 (M2:
  four boots, 225 records, §7y, bundle on the board branch; the pre-registration corrected before the
  numbers were read), F6 prepared (the M1 churn series in the harness, emulator-checked, driver written,
  NOT launched), and R-34 found at the desk (the LSU raises its exceptions and the load unit drops them;
  a stock compliance test fails; both residuals closed by the RTL lane).
* **Runnable next, each gated:** F1's confirming boot (when design A reaches dev — blocked on the
  lead's ruling on collaborator author lines in the scan's range mode); the no-reclamation baseline
  boots (the lead's call; pre-registered, flatness primary, magnitude band 384–391 secondary); the R-34
  confirmation boot on a fixed bitstream (after the RTL lane's fix, sequenced with R-24; the reflash is
  ask-first).
* **Waiting on the lead:** the author-line ruling; METHODS:86 with tab:safety; Q-04; the M1 audit
  order (R-12); R-24 with R-34's fix; the push of `board/e1-s1s2-hardware` (this account gets 403).

## Work items, in order

### H1 — The drivers, lists and chains become portable and go into the repo

New directory `capstone/tests/rtl-smoke/drivers/` with a `README.md` and:

* `board-r1e4.sh` (the parametrised R1 driver: bake → stage → control → N harness invocations →
  control → summary → marker; the live entry point), `board-c6var.sh` (F1's boot: control, cell,
  control, probe), `board-b80s.sh` (the ⑤ `--stats` boot), `board-b78-w2h.sh` (E1's per-arm boot),
  `board-b80a.sh`, `board-b80b.sh`, `board-b79.sh` (the P1 references, kept for the record shape).
* `chain-r1.sh` (from `f5/chain-f5.sh`: the working template for "one chain, N boots, one driver"),
  `chain-m1.sh` (F6, prepared, not launched), `qemu-m1-flowcheck.sh` (the emulator flow check).
* `lists/`: `f5-chase.txt` (45 lines), `m1-diag.txt`, `m1-capacity.txt`, `e4-calibration.txt`,
  `f4-linear.txt` — the invocation lists, with the `run arm series pattern arena extra...` format
  documented in the README.

Portability edits, the same pattern in every script (do not change any measurement logic):

* `R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)` replaces the hard-coded repo path;
  `U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify` replaces the artifacts path;
  `SP=…scratchpad` goes away: the readback host becomes `R1_HOST=${R1_HOST:?}` with
  `R1_HOST_HASH=${R1_HOST_HASH:-2c9e82d101b48160}` (today's value stays the default; a rebuilt host is
  a new hash and needs its own emulator pass — memory `stage-the-same-program-not-the-same-name`);
  `MEMLOCK` keeps its default; the chains take the image/list paths from `drivers/lists/` and
  `${CAPSTONE_ARTIFACTS}/images/`.
* The monitor (`4274268`) and FPGA-copy (`d04bd83`) commit checks, the `qemu-pass/<sha>` record check,
  the lpc hash check and the pre-registration headers stay exactly as they are — they are the gates.
* `pgrep -f 'python3 -m fpga_driver'` inside the drivers stays (it is the runner-overlap guard the
  drivers have always had; the rule against `pgrep -f` is for tool-call shells, memory
  `pgrep-pattern-anywhere-in-command`).
* Every committed script passes `bash -n` and `precommit-scan.sh` (the scan is what forces the path
  edit: `/home/<user>` is on the denylist).

`capstone/sublet/r1/run-r1-qemu.sh` gains one line: on `R1_RC=0` it writes
`${CAPSTONE_ARTIFACTS}/qemu-pass/<sha256 of the image>` (the record the drivers check; today it is
written by hand).

### H2 — The handover document: `capstone/docs/plans/2026-09-15-board-lane-handover.md`

One document the successor reads first, in this order of sections:

1. **Read first**: CLAUDE.md (every hard constraint applies unchanged), the `board-run` and `rtl-sim`
   skills, `docs/ref/HOW-TO-LAUNCH-ON-FPGA.md` (NOT `agent-handoff/`, which no longer exists),
   `docs/state/current-next-step.md` (the 14:45 header), `docs/plans/2026-09-15-sublet-paper-follow-on.md`
   (F1 gate, F5 correction, F6 note), the measurements doc §7r–§7y, ISSUES M-11, C-32, Q-04, R-24, R-34.
2. **The lane's process rules that live only in this session's memory** — distilled, one line each,
   with the memory file name so the lead can copy the memory directory to the apollo server if they
   choose: cite by image hash; the rev-node budget (65,532; one Sublet SQLite workload and one 128 MiB
   arena workload per boot; ≤ 12 region-bearing invocations; ≤ 80 % of the nodes per boot); a
   capability fault in a domain wedges (one expected-fault arm, last); `run-r1-qemu.sh` takes the QEMU
   lock itself (never wrap it in flock); never rebuild the toolchain during a suite; the machine memory
   lock and the lead's one-time private-lock exception; the transcript module and the marker splices;
   the preflight's SIGPIPE lesson (`grep -c`, never `grep -q` under pipefail); ugrep goes silent on
   control bytes (count with python); `pgrep -f` self-match; background tasks end when their work ends;
   a simulation citation needs the artifact hashes at reading time; source-read registry entries name
   the commit; a pre-registered cause number is a source-derived constant; "does not fire" vs "raised
   and dropped" share an observable; the scan's credential-pattern false positive (the word for a credential followed by a colon or an equals sign, in any text); in a shared checkout a peer's
   commits are under your own HEAD; the paper is never edited or pushed by this lane; the push allowlist
   is the user's file.
3. **Where things live on the apollo server** (the successor creates what is missing):
   `~/.claude-c/secrets/{fpga-console-url,name-denylist.txt,push-allowlist.txt}` (the user provides
   them; never echoed, never committed), `~/capstone-artifacts/{unify,images,qemu-pass}/`,
   `PREFLIGHT_ORACLES` (default `/tmp/capstone/ladder-fpga`; regenerate the k800 oracle and every
   `.qemu-pass` by the verify step, never copy them), the paper worktree
   (`git -C capstone/paper-nested-allocators worktree add ~/capstone-artifacts/paper-wt board/e1-s1s2-hardware`
   after the branch is pushed; `make experiments-check` in it).
4. **Artifact lineage, by hash**: build7 `3d6c24a64bceef00` (E4), build8 `55e6a187d52e5cc8` (F4),
   build9 `fdd3029ff0f96680` (M2's four boots), build10 `848887ae81b8c0e3`, build11 `9a01b12a4db639b6`
   (F6, timed), the rr host `2c9e82d101b48160`, lpc `3b93a2b6e2adfa36`, the ⑤/⑥ cells
   (`d61c8bf784f2bbd1`, `c506694f9f6f6889`, `902822a8f303dbaa`, `ceeded2533a74bce`), the C-32 pair
   (`ec061577fb008e18` → `a1f8f2093696d511`), the bitstream `caplifive_r30r31_1bfff7776`, the monitor
   `4274268`, the FPGA copy `d04bd83`. A rebuild on another toolchain is a NEW hash: emulator pass first,
   then cite the new one.
5. **The queue** (the successor's own plan), each item with its gate, driver, pre-registration and
   write-up target:
   * **Q0 — environment proof, before any experiment:** the eleven-line checklist already sent to
     `apollo-board` (repo tip, submodules, toolchain, harness build hash, emulator + lock, buildroot
     built once, module hash, secrets names, scan denylist populated, socketio/requests, docker,
     artifacts + paper clone). Then ONE control-only boot through `board-r1e4.sh` with an empty
     invocation list: bake, load, `RESULT k800 retval=4` twice, marker `done`. Until that passes the
     board is still this session's.
   * **Q1 — F1's confirming boot** when the compiler lane reports design A on dev: rebuild ⑥ at pure
     -O2 (`build-sqlite-silicon.sh`, `SQLITE_OPT_LEVEL=-O2`, no `SQLITE_OPTNONE_FUNCS`), scan
     `movc-cfg-scan.py` (0 integer-only sites other than `renameResolveTrigger`'s PHI residue,
     enumerated), SLT on the -O2 image as the emulator pass with the `movc` density read back, then
     `board-c6var.sh` with the pre-registered counters 5568/37966/32565/37966/5401 and cycles within
     0.1 % of arm C's 1,376,190,813; §7s gains the reading, C-32 → FIXED with the scan as its gate.
   * **Q2 — the no-reclamation baseline** only on the lead's word: `chain-m1.sh` (boot 1 = four
     patterns at the 256-entry capacity, labelled diagnostic by capacity; boot 2 = pressure to the 80 %
     budget), flatness primary (slope × range < 1 % of the mean; last quarter within 1 % of the first),
     magnitude band 384–391 secondary, labelled "no-reclamation baseline, NOT M1", bundle under
     `experiments/results/M1-baseline/`, §7 entry.
   * **Q3 — R-34's confirmation boot** after the RTL lane's fix is synthesised (ask-first reflash):
     the folder's test expects its last arm to enter debug mode (R-24) and the others to trap.
   * **Routine:** a §7 entry per boot, the state doc header, `ISSUES.md` boxes, bundles on the board
     branch with `experiments-check` and the scan run from INSIDE the worktree, commits with `-o` after
     reading `git diff`, pushes gated on the scan's exit status.
6. **The lanes and who owns what today**: paper (the manuscript, `studies.json`, METHODS questions;
   the M2 queried-set series deferred behind M1), compiler (C-32 design A on their branch; the merge
   blocked on the lead's ruling), RTL (R-34's fix with R-24; R-33's third arm; the misaligned-under-
   translation test done), synth (per-module area; leffe), helper, cheri; the apollo-* set as they come
   live. Message by the `ListAgents` names.
7. **This session's role after the handover**: backup, board hands-off; answers questions; runs no boot.

### H3 — Transfer of the unpushed board branch

The paper repository refuses this account's push (403). Two things, both the user's to complete:
add `board/e1-s1s2-hardware` to `~/.claude-c/secrets/push-allowlist.txt` and push it with credentials
that have access; as a fallback this lane writes `git bundle create ~/capstone-artifacts/board-branch-2026-09-15.bundle board/e1-s1s2-hardware`
so the branch can be carried by file. Nothing in the paper repo is edited.

### H4 — State and index updates

* `docs/state/current-next-step.md`: a header line — the board lane is `apollo-board` from the
  moment its control boot passes; this session is backup, hands-off; the drivers live in
  `tests/rtl-smoke/drivers/`.
* `docs/ref/SUBAGENTS.md`: the lane table gains the apollo-* set and the transition note.
* `capstone/tests/rtl-smoke/drivers/README.md` links the handover doc; the handover doc links back.

### H5 — Land and hand over

One commit on dev (H1 + H2 + H4; scan by absolute path before the commit and `--range` before the
push), the bundle file (H3), then the message to `apollo-board`: the handover doc's path and commit,
the reconciliation of their eleven-line environment report against the checklist (any FAIL is theirs
to fix before Q0's boot), and the explicit statement that the board passes to them when their Q0
control boot reads `done`. Copies to paper, compiler and RTL: the board lane's address changes; open
items unchanged.

## Verification

* `precommit-scan.sh` CLEAN on the commit message and on `origin/dev..HEAD`; `bash -n` on every
  committed script; `grep -c '/home/' drivers/*.sh` reads 0; `python3 tests/rtl-smoke/fpga_driver/test_transcript.py` still 8/8.
* The successor's environment report: eleven PASS lines, the harness build hash equal to
  `9a01b12a4db639b6` or explained as a toolchain difference and re-passed under the emulator.
* The successor's Q0 control boot: `RESULT k800 retval=4` twice from a bake on their tree, marker
  `done`, no runner left behind (their task list checked); only then does the state doc name them the
  board lane.
* `make experiments-check` passes in their paper worktree at `a9bafd8`.

## Documents to update (with each item)

* H1: `tests/rtl-smoke/drivers/README.md` (new), `sublet/r1/run-r1-qemu.sh`.
* H2: `docs/plans/2026-09-15-board-lane-handover.md` (new).
* H4: `docs/state/current-next-step.md`, `docs/ref/SUBAGENTS.md`.
* This plan lands beside the handover doc as the record of how the handover was made.
