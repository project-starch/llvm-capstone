# Results-report style

How results-bearing docs in this project are written — history notes, results
summaries, and the numbers-reporting parts of design docs. (Bug-fix/root-cause
trails follow the history/ convention; this is specifically about *results*.)

## Style: article-like, results-first, succinct

- **Lead with the numbers and the finding.** Tables and the key result first; no
  throat-clearing, no narration of the debugging journey.
- **Dense.** Every sentence carries a result or a load-bearing reason. Prefer a
  table plus one interpreting paragraph over prose.
- **One "super-table":** numbers per super-operation *and* a breakdown per
  elementary operation, together. Define **all** measured operations (super and
  elementary), terse and complete.
- **Numbers go in the paper too.** If the work feeds a paper, add the rows / a
  companion table there — never leave a paper table saying "in progress" once the
  numbers exist.

## Do not include

- **Question / task references** ("Q4", "answers Q6", …). The report stands alone.
- **Measurement-limitation hand-wringing** — "fewer iterations than we'd like",
  noise caveats, "needs a 3rd point". State the result; don't apologise for it.
- **Board/FPGA-flaw asides** — flakiness, console drops, reset ceilings, bitstream
  overwrites. These live in the board-stability report. **Exception:** a hardware
  finding that changes the result's meaning (e.g. "revoke is O(tree) because the
  RTL never prunes the revocation list") — that is a result, keep it.
- **Method minutiae and dead ends** — a one-line method note or a separate dated
  history note, not the results body.
