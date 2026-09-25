# tshark rebuilt on dev's compiler — step 1 (2026-09-25, capstone-qemu)

**Question.** The tshark port and every committed tshark result were built with `b7b31421e9fa`, the
shared main-clone build, which predates the C-46 fix and C-47's thread-local support. dev merged
those in `3979abd8e9a3`. Does the port still behave identically when only the compiler changes?

**Answer: yes, on all six pre-registered predictions.** Nothing moved. 36 of 36 fixture cells, all
five stages on three heap arms, and the whole stdout/stderr oracle came out exactly as
`safety-expect.txt` and the pre-registration say they should.

Predictions were written and pushed **before any boot of this step**, in
`docs/history/25-09-2026_12-40-00_tshark-dev-compiler-and-thread-locals.md` (commit `dbc798e988a9`).
`safety-expect.txt` was not edited.

## Verdicts

| | prediction | result |
|---|---|---|
| **P1.1** | every dependency recipe passes its gates; the cross build completes; `diff -r` of the new `xsrc` against the baseline's is empty | **AS PREDICTED.** All seven recipes passed. `cross-build: 0 failed build steps; tshark LINKED`. `diff -rq` of the two `xsrc` trees printed nothing |
| **P1.2** | `build-domain.sh` passes its link gates and negative control on level0, shrink and sublet | **AS PREDICTED.** The control fired on all three. On sublet it names `__capstone_region` as well as `__capstone_hostcall` and `domain_main` — the heap's region, which only that arm has |
| **P1.3** | M1–M5 REACHED and MATCH on all three arms | **AS PREDICTED.** 15 of 15 cells. Each stage returns its own marker (status 101–104, and 0 for M5) with its `TSAPP-STAGE` line |
| **P1.4** | level0 (2 boots) and sublet (3 boots): stdout MATCH on dhcp, dns_port, http, arp, their flips and dns-ooo; ntp DIFFERS; stderr MATCH throughout; every flip fires | **AS PREDICTED.** Including `ntp: stdout DIFFERS (24 diff lines)`, which is the known timezone gap, and all four flip controls firing |
| **P1.5** | one fixture repeat per arm, every run as `safety-expect.txt` predicts | **AS PREDICTED.** 36 of 36 (12 fixtures × 3 arms). Fixture 9, the merged-globals case flagged in advance as the most plausible to move under a codegen change, did not move |
| **P1.6** | level0's M5 peak on dhcp within 1 KiB of the baseline's 28,123,920; sublet's `split + mrev` within 5% of the baseline's 12,596 | **AS PREDICTED, and closely.** level0 `peak_end=28,123,984` — **64 bytes** from the baseline. sublet `split=3953` + `mrev=8643` = **12,596**, the baseline figure exactly |

## What ran

- **The change is the compiler and nothing else.** A private build of dev's tip in the worktree's
  own `llvm/cmake-build-debug`, with the revision pinned to `3979abd8` so worktree commits neither
  change the embedded revision nor re-key the runtime. It was proven object-identical to the
  compiler lane's build of the same commit before this round began. The shared main-clone build was
  not touched. The port's sources and patches are unchanged, which is what P1.1's empty `diff -r`
  establishes.
- **21 boots planned, 29 attempted.** Eight attempts produced `ERROR no section in the log -- the
  image never started`: the guest stalled before the domain ran (ISSUES **I-12**). Each was retried
  and passed. Retries are recorded in `result-lines.txt` and counted as attempts, never as results.
  Section markers are what distinguish this from a domain hang — no section marker means the guest
  never got there.
- **Sizes.** M5 `code_len`: level0 69,741,120 B, shrink 69,741,376 B, sublet 30,488,688 B. The
  sublet arm is far smaller because its heap is a granted region rather than `.bss`.
- `SHA256SUMS` carries the three M5 images, the emulator binary and the guest module's MD5, which
  every boot prints.

## What this does not establish

1. **QEMU only.** No board arm. Every temporal result rests on capstone-qemu untagging a revoked
   capability on load; the deployed silicon forwards it instead (ISSUES **Q-11**).
2. **The thread-local workarounds are still in.** `CAPSTONE_SINGLE_THREAD_DOMAIN` still turns
   tshark's four thread-locals into plain statics, so this round says nothing about C-47's
   lowering in this port. That is step 2, and its predictions are already pre-registered in the
   same history note.
3. **PT_TLS is still empty** in every image here, for the same reason.
4. **One repeat per fixture cell.** N = 1 per arm; the agreement is across 36 independent cells
   rather than across repeats of one.
5. **`ntp` still differs.** The timezone gap is unchanged by the compiler, as predicted; it is not
   fixed by this round.
6. **The toolchain-freshness gate reported rc=2**, "could not fully check", because its default
   target list does not include `llvm-ar`. The staleness check itself still ran and found the
   compiler not stale. This is a gap in the gate, recorded rather than worked around.

## Files

- `result-lines.txt` — every section's verdict line, in the order the boots ran, retries included.
- `heap-counters.txt` — the `TSAPP-HEAP` exit lines that settle P1.6.
- `SHA256SUMS` — the images and tools.
