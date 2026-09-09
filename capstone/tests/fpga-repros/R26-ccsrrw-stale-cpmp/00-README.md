# R-26 — a younger load is checked against the CPMP entry a pending `CCSRRW` is about to replace

> **Status 2026-09-09: fixed in RTL** (`capstone-ariane` `9d8797560`, on `fpga-testing-dev` at `66c4e7517`), **sim-verified**
> at the default memory latency and at a verified 40-cycle latency, lint at the baseline counts, sweep unchanged except
> the predicted cycle deltas. **Bitstream: in synthesis at `66c4e7517`, not yet on the board.** The board arm is the
> board lane's firmware variants D/E (the monitor's CCSRRW-adjacent `fence.i` dropped), filed under `board/` when read.

**Wrong symptom? Read this paragraph first.** This package is the **stale CPMP check**: a capability-CSR write commits
without flushing, and a younger memory instruction runs its permission check against the old entry. It raises nothing
wrong; it fails to raise what it should. `../R27-revnode-orphan-response/` is the hang found while validating this fix —
a flush during a revocation-node query — and is a different defect that the R-26 flush does not cause.
`../S12-wherecode-notcap-operand-vs-memory/` is a nulled operand, not a permission check. `cscratch`/`cepc` are written by
the same instruction but their reader (the domain switcher) runs after commit and was shown unaffected.

## The defect in one paragraph

`CCSRRW` to `cpmp[i]`, `cscratch` or `cepc` takes effect at commit (`csr_regfile.sv`, the `ccsr_we_i` block) and, unlike
every other side-effecting CSR write, never raised `flush_o` (the CCSRRW path clears `csr_we` before the block that sets
the flush). A younger load is not held behind it (the CSR-buffer stall marks only the CSR unit busy) and reads `cpmp`
combinationally in `pmp_data_if.sv`. With an older cache-missing load ahead of the CCSRRW, the younger load's check ran
five cycles before the write landed and the load retired with the data (`sim/ldmiss-events.tsv`, the waveform's event
table; ticks are half-cycles). A divider ahead of the CCSRRW does not open the window (it holds the shared
fixed-latency unit, so the CSR op cannot issue), an ordinary cache miss does. The monitor's `fence.i` after CCSRRW was
the flush the write never triggered.

## The arms (`sim/`, list `sim/testlist_r26.yaml`; kept out of the main suite while an arm was expected to fail)

All M-mode with `mstatus.MPRV=1, MPP=S` so loads are CPMP-checked. CPMP0 starts wide; a narrow entry is written; phase 0
(a load outside the wide entry) proves the check fires, phase 1 is the question, phase 2 (after `fence.i`) proves the
write landed. PASS = 1/1/1 traps; **exit 11 = phase 1 did not trap = the hazard.**

| arm | older op | fence.i | `ef5a8eaf2` | fixed |
|---|---|---|---|---|
| `r26-diag-nodelay`, `-div`, `-div3`, `-fence`, `r26-v2-noolder` | none / `div` | as named | PASS | PASS |
| **`r26-v2-ldmiss`** | cache-missing `ld` | — | **FAIL 11** (1054 cyc at delay 0; 3202 at delay 40) | **PASS** (1103 / 3289) |
| `r26-v2-ldmiss-fence` | cache-missing `ld` | yes | PASS | PASS |
| `r26-v2-cscratch`, `-ctl` | CALL/RETURN context exchange after `CCSRRW CSCRATCH` | | PASS / control FAIL 11 | PASS / FAIL 11 |
| `r26-mret-*` (six) | the monitor's `CCSRRW CSCRATCH; fence.i; mret` shape | | PASS | PASS |

The `r26-v2-cscratch-fence` arm hangs at delay 0 on both trees without the R-27 drain and passes with it: that is R-27.

## The fix (`fix/9d8797560-csr_regfile.sv.diff`)

One statement after the `dom_switch` write chain and before the exception clear: in capability mode, a CCSRRW that
commits its write raises `flush_o`; the controller refetches at `pc_commit + 4` and younger instructions re-execute
against the new entry — CVA6's own idiom for `satp` and friends. `flush_o` is in none of the standing UNOPTFLAT cones;
`ccsr_en` is one, and the fix adds a sink to it, not a feedback edge. Sweep deltas, all predicted: `capsbi-init` +11,
`cpmp-if-check` +3, `interrupt` +5, `s06sec-csr-raw-no-forge` +14 cycles, hashes identical; `cpmp-su-mode` (a TIMEOUT
on both trees, spinning on a fetch check its own S-mode code predates) changes only its cut-off hash.

## Run it

`bash run.sh <checkout> r26-v2-ldmiss` (add `+define+S12_MEM_DELAY=40` as the third argument for the 40-cycle model).

## Records

`docs/history/08-09-2026_22-00-00_r26-ccsrrw-stale-read-demonstrated.md` (the measurement; its correction paragraph
says the waveform was delay 0), `docs/history/09-09-2026_16-00-00_r25-r26-fix-cycle.md` (the fix cycle),
`docs/ref/csr-write-serialisation.md` (what does and does not serialise against a CSR write). Registry R-26.
