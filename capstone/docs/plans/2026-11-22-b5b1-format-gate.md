# B5b-1 step 1b (the Sublet memory format in the core): synthesis-only gate, board lane copy of the pre-registration

Status: pre-registered 2026-11-22, before the launch. This is Line B option C, as the lead ruled. The keys are the rtl
lane's (land/rtl-b5b1 365f5caeee95, `docs/plans/2026-11-22-b5b1-format.md`, "Step 1b's synthesis-only gate"), copied
unchanged.

**Tree, read at source.** capstone-ariane b5b1-format 4c576521d.
- Its base 1c5d31474 (the R-75 (d) tree) is an ancestor.
- `core/` and `corev_apu/` are identical to bd686c6a4's (`git diff --stat` is empty).
- Against the base, nine files under `core/` change, +186 / -20:
  - the dom_switcher .anvil and .anvilh;
  - decoder.sv, ex_stage.sv, ariane_pkg.sv, the config package, load_store_unit.sv, load_unit.sv and store_unit.sv.
- Outside `core/`, only `verif/` changes (two directed tests, a testlist and the lint REF). No submodule pin moves.
- The config package sets `define CAPSTONE_SUBLET_FMT` unless `CAPSTONE_IDS55_FMT` is defined. No file under
  `corev_apu/fpga` and no Makefile names `IDS55_FMT`, so the FPGA flow builds the Sublet format.
- `*.anvil.sv` is git-ignored and generated in the run. The R-75 (d) gate's switcher SV was written at 08:13:31Z, after
  its 08:08:19Z launch. This gate also records the generated switcher's hash against the base's, and its whole-word
  `snap` count (the base reads 0).
- Lint on bd686c6a4 is PASS by the rtl lane's numbers. UNOPTFLAT is 39, with the names identical. UNUSEDSIGNAL was
  re-baselined to 1661; all of that change is in the generated switcher.

**Gate** (`twc/b5b1/post-b5b1.sh`, graded by `twc/b5b1/grade-b5b1.py` against the R-75 (d) gate,
`<mnt-hdd>/synth/r75d-1c5d31474-gate-out`). The grader was dry-run twice:
- the base against itself: PASS 4/4, every delta +0;
- B5a's gate as trial: FAIL on mapping (+1,814 MB) and core LUT (+2,804).

| key | prediction | if missed |
|---|---|---|
| mapping peak | within 10 % of 5,051.6 MB, inside the 40 GB guard | guard kill: partition read, no retry |
| loops | 0 by every probe, each control firing | named |
| core LUT / FF | 162,982 ± 2,500 / 54,345 ± 2,000 | per-module census |
| dom_switcher LUT | reported against the base. The 11 inline range tests are the only new logic there. Above +1,000, the rtl lane moves the snap computation to ONE SV-side test before any full build | |
| load_unit / store_unit / ex_stage LUT | reported. The decoder is reported in id_stage_i | |

No full build follows: the genesis change is owed first.
