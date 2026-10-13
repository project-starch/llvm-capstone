# The system-allocator cases: baseline and CheriBSD readings -- 2026-10-11

The seven cases whose object ends in libc, through `mrb_free` or a moving `mrb_realloc`: 01, 03,
06, 09, 10, 11 and 17. Pre-registered in [PLAN.md](PLAN.md) (pushed as 2af056837581 before either
run), with the fault sites from host ASan. Both runs on p13, one boot each. Result lines only;
the raw console captures are not committed.

## Baseline without temporal protection: `bounds/`

The 2026-10-06 `sysalloc-bounds` image (`mruby-bounds.dom`, sha256 `0f208270...`): the Capstone
application domain's first-fit heap, each allocation bounded, nothing revoked. Same platform as
that day (hashes in `bounds/inputs.json`), shipped to p13. Case 11 runs `capi.c` linked against
that build's `libmruby.a` (`capi-11.dom`, `e98e3181...`; the build's SDK pointed at the dev linker
script of 2026-10-06, 1b7d47d39732, its own worktree being gone; segment layout as the arm image).
Controls in the same boot: `smoke.rb` printed `SMOKE_DONE`, 40 and 500 frames returned.

| case | outcome | fault in | reaches the access |
|---|---|---|---|
| 01 | cause 24 | `mrb_vformat`, not its site (`mrb_vm_exec`) | no |
| 03 | completes, `["PASS"]` | | yes |
| 06 | cause 5 | `kh_get_set_val`, its site | yes |
| 09 | completes, `["PASS"]` | | yes |
| 10 | completes, wrong answer | | yes |
| 11 | `CASE11 ready`, then cause 24 | `gc_mark_children`, its site | yes |
| 17 | cause 24 | `mrb_func_basic_p`, not its site (`mrb_hash_pat_values`) | no |

Against the predictions: 03, 06, 09 and 10 as predicted. 01 and 17 fault where predicted in kind
(cause 24) but not in place: outside the access, so by the rule fixed beforehand they leave the
denominator. The first boot's smoke control was killed by the 25 s case watchdog (no reading);
the second gave controls a 600 s budget and every case the same status, cause and pc as the first.

## CheriBSD: `cheribsd/`

The 2026-10-06 interpreter (`5784fa99...`, relinked byte for byte from its objects) and case 11's
driver, each plain and with the quarantine probe (`fa283f76...`). Revocation at the default
(sysctls read back: on, asynchronous, not every free). A fault's place is the console's CHERI
fault record (`machdep.log_user_cheri_exceptions=1`), its `sepcc` resolved against the image.

Controls in the same boot: eval printed 42; the 2026-10-06 `d40.rb` and `d500.rb` returned; the
platform's revocation control took a tag fault at its labelled load (`read_probe+0`) after a
forced sweep; the probe's sweep counter read 6 under a churn of 400,833 allocations and 0 under a
quiet run.

| case | plain run | probed: frees / quarantined / reissued while quarantined / sweeps | reading |
|---|---|---|---|
| 01 | tag fault in `mrb_vformat`, not its site | (faults) | no reading |
| 03 | completes, `["PASS"]` | 1238 / 1238 / 0 / 0 | held |
| 06 | length (bounds) fault in `kh_get_set_val`, its site | (faults) | rejected, by bounds |
| 09 | completes, `["PASS"]` | 1459 / 1459 / 0 / 0 | held |
| 10 | completes, wrong answer | 1506 / 1506 / 0 / 0 | held |
| 11 | `CASE11 ready`, `CASE11 completed 30` | 869 / 869 / 0 / 0 | held |
| 17 | completes, wrong answer | 1994 / 1994 / 0 / 0 | held |

Against the predictions: 01 faults where the baseline does, not at its site; 06 at its site, by
the stale capability's bounds rather than a sweep; the rest as predicted. "Held": every free of
the run went into the quarantine, no allocation came back while quarantined and no sweep
completed, so no freed block was handed out again and the stale pointer read the dead object
only. The counters are process-wide and do not name the stale block; they establish that none
was reissued. The probed runs ended as the plain ones did.

Two boots before this one scored nothing: the first stopped when a 121 MB uncompressed kit
exceeded the guest runner's 120 s copy limit, the second when the runner's own first depth
control (`fmt-d40.rb`, the same recursion printed through `String#%`) faulted. That script now
runs after the controls as an observation: a CHERI length fault in `mrb_str_format`, in a correct
program. Not followed up here.

## For the security table

Scored by the rule in PLAN.md, with the Sublet column from virtual mallocng
(`results/2026-10-11-virtual`, `-virtual-capi`: cause 25 at each case's declared site):

| case | reaches (baseline) | CheriBSD | Sublet |
|---|---|---|---|
| 03 | yes | held | cause 25, `mrb_str_format` |
| 06 | yes | rejected (bounds) | cause 25, `kh_get_set_val` |
| 09 | yes | held | cause 25, `mrb_iv_foreach` |
| 10 | yes | held | cause 25, `mrb_struct_equal` |
| 11 | yes | held | cause 25, `gc_mark_children` |
| 01, 17 | no | -- | -- |

System allocator: 5 cases, CheriBSD 5 (four held, one rejected by bounds), Sublet 5.
