# Pre-registration: naming which of resolve()'s five refusals fires

Written and pushed BEFORE the QEMU control run and before the board boot.

## Why this instrument exists

`vm` reports a bare **139** when it fails. That signal cannot say *which* refusal inside
`resolve()` produced it, and the attribution currently on the table — the `!same` page-identity
check — is a **structural deduction from the source, not a reading**. Nobody has read a value.

Three competing explanations for r8's terminal cause-15 store have already been refuted by
measurement (an A/D-bit loop; a guest-kernel-version difference; `svadu`, refuted in
`RESULT-svadu-false-refutes-the-synthesis.md`). The `!same` deduction is the only survivor, and
it survives on reasoning alone. This change makes it readable.

## What changed

Every terminal return in `resolve()` now names itself at `pr_warn`. Five sites:

| site | condition | errno |
|---|---|---|
| `vma` | no VMA, wrong start, `required` bit absent, file-backed, or `VM_SHARED` | `-EACCES` |
| `pin` | `pin_user_pages_remote` did not return 1 | `-EFAULT` |
| `anon` | `!PageAnon` or `page_mapcount != 1` | `-EACCES` |
| `remap` | `a->pages[index] != page` — the `!same` path | `-EACCES` |
| `arena` | no lazy arena covers the address | `-EFAULT` |

Each line also carries `prev(addr, cause, rep)`, which costs nothing because those fields are
already on the thread for the `-ELOOP` bound, and which is the discriminator: **prev addr equal to
this addr with prev cause 13 and cause 15 here is the read-then-write COW signature.**

Three deliberate design constraints:

- **No print per fault.** A measured 426 faults per step would be ~40 KB of UART per step and
  could perturb or wedge the boot the print exists to observe. Every one of these five paths is
  terminal, so the bound is **at most one line per step** with no rate-limiting needed — and
  rate-limiting was rejected precisely because it could suppress the one line that matters.
- **`pr_warn`, not `pr_info`.** The board console drops `KERN_INFO`; that is what voided r7's
  diagnostic entirely, because an invisible print and a print that never fired read identically.
  Verified in the built object: all six new format strings carry the `<4>` KERN_WARNING prefix.
- **`page_to_pfn`, not `%p`.** `%p` is hashed, so two values are comparable for equality within
  one boot and for nothing else.

## The must-fire control, and why it was needed

`init()` now emits one line per `insmod` at the same level, carrying `CV_DIAG_TAG` and
`supervisor_slots`.

This is not decoration. Searching every serial log on this host — QEMU and board, back through
2026-10-16 — finds **no `capstone_vm:` message line, ever**. (The string `capstone_vm` does occur,
as the `.ko` filename in the `insmod` command, which is what proves the search can fire.) So
before this change an absent diagnostic was uninterpretable: "the level is wrong" and "the path
did not execute" were the same reading. With the banner they are distinguishable:

- banner **absent** → the level is being dropped; any absent diagnostic below is **void**;
- banner **present**, no site line → the path genuinely did not fire.

## Predicted reading per hypothesis — the go/no-go

Written before the run. Per CLAUDE.md this is a **decision input**, not documentation: a build
whose predicted reading is uninformative does not go.

| if true | site | predicted values |
|---|---|---|
| **COW on a write to a read-touched page** (live hypothesis) | `remap` | cause 15, `pfn_old != pfn_new`, `vm_flags` has `VM_WRITE`, `prev(addr == this addr, cause 13)` |
| mprotect/VMA state not visible to the module | `vma` | `found 1`, `flags` without `VM_WRITE` (`need 0x2`), `file 0` |
| the page stopped being exclusively-mapped anon | `anon` | `mapcount != 1` or `anon 0`, with `pfn` |
| GUP itself refused | `pin` | `pin_user_pages_remote=` a negative errno or 0 |
| the address left every lazy arena | `arena` | the address, and no arena |
| **none of the five** (banner present, no site line) | — | **the store is not failing in `resolve()` at all**; r8's 139 comes from elsewhere and the `!same` deduction is **refuted** |

That last row is why this build is worth a cycle: it is the outcome that redirects the
investigation rather than confirming what is already suspected, and the banner is what makes it
readable instead of ambiguous.

## Compile-time gates, already passed

- `-Wformat -Werror=format` on both arms: **0 format diagnostics**. The five format strings are
  therefore argument-checked by the compiler, not by inspection.
- **RVC = 0** on both arms. A compressed instruction in the `.ko` hangs the FPGA at `insmod`.
- All five distinct format strings present in the board object, each with its site-specific tail,
  plus the banner; `CV_DIAG_TAG = resolve-sites-2`.

Module hashes:

- board (linux-6.4.14, flat): `52adaac58db98b9f6bdf15d4e832b1c7b9e70b58f1d51debbad1491c5197b66f`
- QEMU (linux-custom, flat): `e8e65536cd342609...` (rebuilt with `pc`; see the durable `SHA256SUMS`)

Durable board artifact: `diag-r11/` on the shared scratch volume, with its own `SHA256SUMS`.
Each line also carries **`pc`** (`t->frame[4]`, the slot `CV_STEP` reports as `r.pc`), so the
faulting instruction is named rather than inferred.

## QEMU control run — predicted, before running

Same configuration as the immediately preceding run (the `svadu=false` wrapper, v9 images,
`supervisor_slots` default 1), so the only variable is the diagnostic. The change is
observation-only on every non-terminal path, therefore:

- **`34 / 41`**, with exactly the seven false checks `threads, vm, heap-threads, threads_output,
  vm_contract, heap_threads, shared_heap_pages_released`;
- the same `EXITS` line, and `CAUSE15=0 CAUSE25=0`;
- the **banner present** in the log — which is what promotes "no site line fired" from an
  assumption to a reading;
- **any** change in the pass count or the false set is a **defect in this change**, not a finding.

Note on the QEMU side specifically: `vm` exits **1** under QEMU, where the store *passes*, so the
`remap` site is **not expected to fire there**. The QEMU run is a control for the banner, the pass
count and the line format — not a test of the hypothesis. Only the board can produce the terminal
store.

## Handoff note for the board lane

`run.py --skip-recycling` omits **both** `sparse-churn` and `sparse-stale`, and changes the
`fault_causes` expectation at `run.py:222` (the trailing `24` is dropped). It is therefore not
"sparse-churn moved to last" — it loses a row. Two invocations, in this order:

1. `--skip-recycling` — reads `vm`'s line and everything else, with no wedge risk;
2. `sparse-churn` alone, last, because it takes the core with it and everything after it is lost.

Expectation delta between the two is exactly that trailing fault cause `24`.
