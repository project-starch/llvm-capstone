# Native kit files kept in the repository

Kits (`t3-smoke/bkit*`) are local directories with a SHA256SUMS each; a running or queued chain's kit is never edited.
A new kit copies the files here.

## spec-classify.py (2026-11-23)

The kit classifier until bkit108 read only the module route's markers, so every native-route transcript was
classified "NO RESULT: no @@SPEC-DONE". Native readings were taken from run-stage.py read (perf/natstage/read-one.sh)
instead. This version reads both formats. On the native route: `@@SPEC-STAGE`, `@@SPEC-END <i> <kind> <arm> <rep>`, one
hash block per arm, and `@@SPEC-DONE <bench> <size>` or `@@SPEC-DONE <n>`.

Checked on recorded transcripts (out-nat-* under spec/board/):

| Case | Transcript | Reading |
|---|---|---|
| native, 1 command | p1-frcdkmsspa3-trainb-b1 519.lbm_r-train | PASS |
| native, 2 commands | p1-l1d48full-b2 500.perlbench_r | PASS |
| native, gen + run + 3 validations | p1-frcdkmsa3-train2-b2 525.x264_r-train | PASS |
| native, two commands rc 139 | p1-frcdkmsa3-train2-b2 500.perlbench_r-train | FAIL, rc 1 |
| no @@SPEC-DONE (edited copy) | perlbench above | NO RESULT, rc 2 |
| one @@SPEC-END removed (edited copy) | perlbench above | MALFORMED, names the open command |
| wrong DONE count (edited copy) | perlbench above | MALFORMED |
| pgbench-derived stage, `@@SPEC-DONE 4` | twcf-hpmnest-pg-b1 p2-stage | PASS (the old classifier: MALFORMED) |
| --twin, same transcript | perlbench above | identical, PASS |
| --twin, one hash changed | perlbench above | DIFFER, FAIL |

No module-route transcript in this shape was found among the board outputs, so the old branch is kept but unexercised.
