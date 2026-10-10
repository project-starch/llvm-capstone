# The CheriBSD arm: predictions

Registered 2026-10-11, before any corpus case ran on this guest. A result that differs is a
finding; this file keeps the prediction that was made.

## The arm

Perl 5.36.3 for CheriBSD purecap from `ports/perl/cheribsd/build.sh`, now linked dynamically
with `-cheri-tgot-tls` (the SDK's own purecap configuration). The 2026-10-06 arm could not use the
stock image: its loader refused the dynamic binary ("Traditional TLS not supported"), static TLS
was broken, so it ran on another platform with revocation selected per process. That flag
removes the constraint: on the stock purecap image (e7470361, booted from snapshot on p13, the
image the CPython arm used) the dynamic interpreter evaluates `6*7` with
`security.cheri.runtime_revocation_default: 1`, system-wide, as the platform ships it.

The runner (`runners/cheribsd/run-cheribsd.sh`) preloads the fault reporter and the quarantine
probe (`tools/cheribsd/build-helpers.sh`) and runs every case twice: revocation on (the default),
and off for that one process (`_RUNTIME_REVOCATION_DISABLE=1`). Its controls refuse the run unless
the self-test reports `PROT_CHERI_BOUNDS` and `PROT_CHERI_TAG` through the preload, the same
`revoked` read completes with revocation off (the knob is live; checked on this guest: the free
then never enters the quarantine), `6*7` evaluates with the probe reporting, and the harness
loads.

## How a row is read for the table

* **System allocator** (03, 04, 06, 07, 10): CheriBSD earns use-after-reallocation credit when
  either the revocation-on run ends on SIGPROT `PROT_CHERI_TAG` and the revocation-off run does
  not end on the same fault (the same si_code at the same address: revocation made the
  difference, possibly by stopping the access earlier than the off run's crash), or the
  revocation-on run ends without a fault and the probe reports every free quarantined and none
  reissued while quarantined. A case whose on and off runs end on the same fault earns nothing:
  that fault is not revocation's. This is the counterpart of the Capstone rule, where cause 25
  names a revoked lifetime by itself and a cause-24 fault is not counted.
* **Nested allocator** (01, 02, 05, 09): an SV head is recycled on `PL_sv_root` and never passed
  to `free()`, so the quarantine never holds it (the 2026-10-06 probe on 05: 2,983 frees, all
  quarantined, none reused, and the head not among them). These earn no credit whatever the
  probe reports, unless a revocation-on tag fault that the off run lacks shows otherwise.

## Predictions

From the 2026-10-06 matrix (revocation off / on, then per process on another platform):

| case | revocation on | revocation off | credit |
|---|---|---|---|
| 01 | SIGPROT | SIGPROT, the same fault | none (not revocation's) |
| 02 | Perl's own panic | the same | none |
| 03 | completes | completes | the trigger does not reproduce |
| 04 | SIGPROT | SIGPROT, the same fault | none (not revocation's) |
| 05 | completes | completes | none (nested) |
| 06 | SIGPROT | SIGPROT, the same fault | none (not revocation's) |
| 07 | SIGPROT, `PROT_CHERI_TAG` | completes | UAR |
| 08 | SIGPROT, `PROT_CHERI_BOUNDS` | the same | not temporal |
| 09 | SIGPROT | SIGPROT, the same fault | none (not revocation's) |
| 10 | SIGPROT, `PROT_CHERI_TAG` | completes | UAR |
| 11 | completes, wrong bytes | the same | none |

So the predicted CheriBSD cells: system allocator 2 of the 4 (07, 10); nested 0.

## Amendment, after the first run: patch 0010 and a rerun

The first run (`first-run/`, interpreter built with patch 0009) passed every control. 04 and 06
both faulted at one instruction in `Perl_sv_2uv_flags`, revocation on and off: the `builtin::`
descriptor round trip through an integer, which every program loading `overload.pm` hits
(`results/2026-10-11-virtual/PREDICTIONS.md`, amendment). The other rows: 01 and 09 SIGPROT
`PROT_CHERI_TAG` at the same instruction on and off (Perl_pp_iter, Perl_newATTRSUB_x); 02 Perl's
panic; 03 and 05 completed; 07 completed with a wrong result and every free quarantined, none
reissued; 08 SIGPROT at one instruction on and off; 10 completed with [PASS] and every free
quarantined, none reissued; 11 completed. The rerun uses an interpreter rebuilt with patch 0010,
the same runner, rules and predictions.
