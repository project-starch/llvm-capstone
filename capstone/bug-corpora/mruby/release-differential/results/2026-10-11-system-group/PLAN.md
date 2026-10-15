# The system-allocator cases: baseline and CheriBSD readings -- pre-registration, 2026-10-11

Written and pushed before either run. The cases are the seven whose object ends in libc, through
`mrb_free` or a moving `mrb_realloc`: 01, 03, 06, 09, 10, 11 and 17. The virtual arms have
already read all seven (cause 25 on virtual mallocng, in `results/2026-10-11-virtual` and
`-virtual-capi`); what is missing for the security table are a baseline without temporal
protection that reaches each access, and a CheriBSD reading that says whether the freed block
was rejected, held or reissued.

Fault sites for 01, 03, 06, 09, 10 and 17 are declared in each `case.json` in the same commit as
this file, from host ASan at the pin (the access ASan reports as heap-use-after-free). Case 11's
were declared in 2358c168ca73.

## Runs

1. **Baseline**, `probe/run-bounds-system.sh`: the 2026-10-06 `sysalloc-bounds` image
   (`mruby-bounds.dom`, sha256 `0f208270...`) on the platform it ran on, shipped to p13, plus case
   11's `capi.c` linked against that build's `libmruby.a`. Controls first: `smoke.rb`, 40 and 500
   frames.
2. **CheriBSD**, `probe/run-cheribsd-system.py` on p13: the 2026-10-06 interpreter (sha256
   `5784fa99...`, relinked byte for byte from its objects) and case 11's driver, plain and with
   the quarantine probe (`probe/build-cheribsd-extra.sh`). Controls first: sysctls at the
   default, eval, 40 and 500 frames, the platform's revocation control, and the probe's sweep
   counter under a churn and under a quiet run.

## Predictions

| case | baseline (bounds only) | CheriBSD plain | CheriBSD probed |
|---|---|---|---|
| 01 | cause 24, as 2026-10-06; in `mrb_vm_exec` | SIGPROT, as 2026-10-06; in `mrb_vm_exec` | no counters (a faulting process runs no destructor) |
| 03 | completes, `["PASS"]` | completes, `["PASS"]` | every free quarantined, none reissued while quarantined, sweeps 0 |
| 06 | cause 5, as 2026-10-06; in a site | SIGPROT; in a site | no counters |
| 09 | completes, `["PASS"]` | completes, `["PASS"]` | as 03 |
| 10 | completes, wrong answer | completes, wrong answer | as 03 |
| 11 | no firm prediction: `CASE11 ready`, then completes, or a fault in `gc_mark_children` | completes, `CASE11 completed 30`: the freed stack stays quarantined and intact | as 03 |
| 17 | cause 24; in `mrb_hash_pat_values` | completes, wrong answer | as 03 |

## How each reading is scored (fixed now)

- **Baseline reaches the access** when the run completes, or faults inside the case's
  `fault_sites` (the access itself, on storage the heap reissued; for 11, after `CASE11 ready`).
  A fault anywhere else, or a run without its control, means the case leaves the denominator.
- **CheriBSD is credited (UAR)** when the plain run faults inside the `fault_sites` (the access
  was rejected), or when it completes and the probed run shows every free quarantined, no
  allocation returned while quarantined, and no completed sweep (the quarantine still held the
  block). A fault outside the sites is not credited.
- **Sublet is credited (UAF)** when virtual mallocng faulted inside the `fault_sites`, already
  measured.
