# Temporal-safety security comparison: Capstone, Sublet, CheriBSD, PoisonCap

Base: `ports/9-poisoncap-ffmpeg`, commit `69925e284db7`, merged with `origin/dev`
`8abc0545f0dd`.
Work branch: `ports/10-temporal-security`.

## Objective

Establish what each of the four implementations actually denies an attacker over
an object's lifetime, and where each one stops. The existing port results answer
"does the protected arm fault where the unprotected arm does not" for accidental
misuse. This lane asks the adversarial question instead: given a retained alias
and a chosen moment, what still works.

The unit of comparison is an attempt, not a mechanism. A row of the matrix is
something an attacker tries; a cell is what one arm does when they try it. An
arm that has no implementation of a row gets an empty cell, not a predicted one.

## The four arms as they exist here

| Arm | Temporal mechanism | Acts at | Already built |
|---|---|---|---|
| Capstone | revocation capability (`revoke`, `mrev`; opcode `0x5b`) | the contract point, synchronously, O(1) | QEMU and FPGA; [A1 alias matrix](../../ports/ffmpeg/buffer-pool/security-tests/capstone/README.md) |
| Sublet | linear slot discipline over Capstone ([`sublet.h`](../../sublet/README.md)) | lender/borrower boundary of one allocator's five operations | per-port patches; replay arms in the FFmpeg/PostgreSQL/CPython/Whisper ports |
| CheriBSD purecap | spatial by construction; temporal only by quarantine-driven revoker sweep | `free()` plus a deferred sweep | [four allocator libraries and runner](../../ports/common/host/cheribsd/README.md); spatial replay arms |
| PoisonCap | per-lease poisoning with synchronous revocation on a CheriBSD base | lease return, with sweep before reuse | [pilot adapter and controls](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md) |

Two of these are ours and two are not, which is exactly why the matrix has to be
written before the runs: an evaluation whose rows were chosen after the results
were seen is a ranking, not an evaluation.

## Attacker classes

* **A-1 — buggy consumer.** No intent; the lifetime contract is violated by
  mistake. Every existing corpus and replay result sits here.
* **A-2 — hostile consumer.** Holds the object legitimately, then retains
  aliases deliberately and picks when to use them. Partially covered today by
  the A1 alias-scatter fixture, which retains copies in five locations and does
  not clear them.
* **A-3 — untrusted nested manager.** A sub-allocator that receives authority
  from a parent manager and tries to keep it past the parent's revocation, or to
  hand it onward. Nothing here covers this yet; the existing hierarchy work is
  trusted adaptation.
* **A-4 — reuse-controlling attacker.** Influences allocation sizes and order to
  decide what lands at a freed address and when a deferred sweep runs.

State the class on every result. A cell proved under A-1 does not transfer to
A-2.

## Attack matrix

Rows are attempts. The per-arm expectation column is what we predict *before*
running, and it is recorded so that a wrong prediction stays visible.

| | Attempt | Turns on |
|---|---|---|
| **T1** | Dereference a retained alias after return, before any reuse | whether invalidation happened at all |
| **T2** | Dereference it after the same address is reissued | whether the old alias is distinguishable from the new owner's |
| **T3** | Same, from a location a sweep may not reach: register, callee-saved spill, global, heap object, list node, a sibling arena, kernel-held or file-backed storage | sweep completeness |
| **T4** | Use the alias inside the deferred-revocation window | the window's existence and length, measured rather than asserted |
| **T5** | Violate the lifetime contract with no `free()` anywhere | whether the mechanism is bound to the allocator's free path |
| **T6** | Land the object in a granularity gap: sub-granule sizes, padding, compressed-bounds slack, partially poisoned storage | protection granularity |
| **T7** | Attack the mechanism: retained manager authority, replayed or forged revocation handle, double revoke, revoking the *new* owner, and a revocation call that fails while reuse proceeds | whether the enforcement path is itself authority-checked |
| **T8** | Race the alias against the sweep, the poison and the detox | atomicity of the transition |
| **T9** | Hierarchy: deep child after ancestor revocation, a child that escapes its parent's subtree, a child retaining a senior handle | scope of one revocation |

T4 and T5 are where the arms are expected to separate, and both must be
*measured*: T4 as a window in the arm that has one, T5 as a program that never
calls `free` and still violates its contract. T7 is the row most likely to
produce a negative result about our own arms, which is why it is in the matrix
and not an appendix.

## First measured cell, and it is a loss

The FFmpeg corpus supplies seven cases of the sharing taxonomy's class 3,
*reuse-not-free*: one pooled buffer, two holders, the one that kept it writes
into it again. Nothing is freed, the pointer stays tagged and in bounds, only
the identity of the data changes.

Probe case 39 runs that shape in a domain. **Mode 2 completes.** The port's
Sublet adapter hooks `sublet_take` at pool issue and `sublet_give` at pool
return; nothing hooks `av_buffer_ref`, so a second reference is not a borrow and
there is no return to revoke. The fixture verifies its own premise first —
reference count 2, `av_buffer_is_writable` false — so the shape is real and the
write lands regardless.

This is T7 territory reached from the other side: not an attack on the
enforcement path, but a class the enforcement path does not reach. It is the
first cell in this matrix that our own arm loses, it was found by building
cases rather than by reasoning about the model, and covering it needs Sublet's
borrow primitive rather than its revocation primitive.

## Evidence rules

These are the A1 fixture's rules, restated because this lane will produce cells
for systems we did not write.

1. **Every blocked cell needs a matched control that succeeds** at the same
   access site with protection off. A fault with no passing counterpart proves
   the fixture is broken, not that the attack failed.
2. **A fault counts only with cause, exact labelled PC, and a completed-setup
   marker.** Timeouts, arbitrary crashes and setup assertion failures are not
   successes.
3. **A bounds fault is not temporal evidence.** Separate it from a lifetime
   fault explicitly; the CheriBSD spatial arm exists to make that distinction
   visible.
4. **Show the reuse.** "The mechanism denied it" and "the allocator never
   reissued that address" look identical in a log. Every T2/T3 cell records the
   observed reuse at the same address.
5. **Retain failed attempts**, including platform failures that stopped a run
   before it reached its oracle.
6. **No timing or performance claims from this lane.** A denial-of-service
   result (T7's revoke-the-new-owner case) is reported separately from
   confidentiality/integrity results; they are different claims.
7. **Negative results about our arms are first-class.** This lane is not a
   search for a Capstone win, and a row that Capstone loses is the most valuable
   thing it can produce.

## First steps

1. Fix the matrix as a machine-readable file plus a generated status table, with
   every cell starting as "no implementation". No runs yet.
2. **Capstone/Sublet:** the A1 fixture already covers T1–T3 across five alias
   locations with its no-revoke control. Extend it to T5 (contract point without
   `free`) and T9 (deep child, escaped child, retained senior handle) under the
   same oracles, and record the class as A-2.
3. **CheriBSD:** T4 needs the quarantine window measured on the existing purecap
   runner. The spatial arm's stale access *succeeding* is the control, not a
   defect; record it that way.
4. **PoisonCap:** T6 against the pilot's 16-byte poison granularity and its
   snapshot/restore path, and T7 by injecting a failed revocation to check the
   adapter stops before reuse, which its design requires.
5. Compose the cross-system table only after each arm has its own controls
   passing. An arm without controls contributes an empty row.

## Non-claims

This plan does not assert that the four arms provide comparable guarantees, that
their protection scopes are equivalent, or that a compiling library implies
inner temporal protection — the CheriBSD port guide is explicit that it does
not. It makes no performance comparison and produces no ranking. Reconstructed
third-party platforms are evaluated as reconstructed and pinned here, which is
not the same as their authors' deployed configuration; the PoisonCap pilot's
disabled guest libc-revocation default is part of the configuration under test
and must appear beside every PoisonCap cell.
