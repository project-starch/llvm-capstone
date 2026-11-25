# B5b-2 4(b): interruptible CREVOKE in the RTL (parts 1 and 2), 2026-11-25

The design is llvm-capstone `capstone/docs/plans/2026-11-24-interruptible-crevoke.md` (spec-silicon d3a4a2259bd8);
the reference is capstone-qemu da49602d76 (`cap_rev_kill_step`). RTL: capstone-ariane branch `b5b2-crevoke4b`, on
904cd9c6b (the B5b-2 next bitstream, on boards 2 and 3).

## What was built

- **Part 1** (e3ba8aed0): CREVOKE finds the run's last record t with the forward scan, then releases the run from its
  END backward in chunks of KCHUNK = 64. Each chunk is checked walking back from t before anything in it is released:
  each member live, deeper than the operand, with a nonzero prev whose record names it as next. A failed check is 29,
  with earlier chunks left released and the header written. The chunk is then released as the walk reaches it and
  spliced out (p.next = stop, stop.prev = p, each re-read first as the reference re-gets them).
  - After a chunk, a pending interrupt ends the operation with status 2 (PARTIAL).
  - ex_stage shows the DYN unit status 0 and marks that CREVOKE's result with the R-43 replay cause, so commit
    re-fetches the same pc and the decoder takes the interrupt there (epc = the CREVOKE). The re-executed CREVOKE
    starts again from its checks.
  - The predicate (cva6.sv) is the decoder's own interrupt selection (without RVH), plus a debug request.
  - CRENEW is unchanged (the forward pass, list order, one execution), as in the reference.
  - Records leave in reverse list order, then the operand, so later CDERIVEs take slots in that order.
- **Part 2** (710998447): the saved position, option A under the semantics owner's ruling of 2026-11-25. The contract
  clause: between a stop and its restart, the operand's run changes only through lifetime instructions; a program
  that writes such a record any other way gets an unspecified result bounded by B1' (only records that pass the chunk
  check are released), B2 (it ends, causes 24-30) and B3 (writes only table words).
  - A PARTIAL keeps {operand id, namespace key, t, t's generation, stop}. A restart of the same id in the same
    namespace resumes from t if t re-reads live at that generation, deeper than the operand, with stop as next, and
    stop names t as prev; otherwise it scans afresh.
  - It is forgotten by CDERIVE, CRENEW, CSMINT's record, a CREVOKE of another id, the CREVOKE's end, an srevroot
    change, a CSTRAW dispatch, debug mode and a debug request.
  - No scan-phase stop (the ruling; a single kept id livelocks two contexts), so the first execution's scan is O(n).
- **Fixes:** ca1ea676d (the claim-auditor's corrupt-table case: the splice between chunks wrote stop from a stale copy;
  reproduced, then fixed by re-reading p and stop); bf8fc0cfb (the dead register that fix left; the lint's one name).

## Readings

- **Unit against the reference**: `sublet-life/crevoke-chunk-diff.py`, `sublet-life/crevoke-sv-diff.py` and
  `sublet-life-diff.py` (capstone-ariane, beside the unit testbench).
  - Chunk shapes (chain, fan, random trees at 1, 2, K-1, K, K+1, 2K+1, 3K+2 and 1,000 records; CDERIVE / CREVOKE between
    executions; a stop never restarted; broken links; the auditor's cross link): equal at K 64 and 256, and with a stop
    after every chunk at K 1, 7 and 64. The no-back-link mutant differs on the wrong-link case only.
  - Saved position, 12 in-contract cases at K 1, 7 and 64: equal, each restart on its expected path (resume; fresh
    after every item of the invalidation list; miss after a store into t). Out of contract, a depth store into the
    run: B1'-B3 hold.
  - Random streams: 82/82.
  - Latency at the unit (2-cycle memory, interrupt held, K 64): restarts 1,098 cycles on part 2 against 25,284 on part 1
    (4,096 records); the first execution scans 394,270 cycles at 65,532 records.
- **Core** (Verilator, memory delay 12, `+R43_TRACE`):
  - 50 in-tree Sublet tests on part 1, against 904cd9c6b's three core files: 47 identical in value (two ctxport arms
    +12 cycles), and b5b2-crevint's three arms as pre-registered in value and checksum. The base reads the list-order
    ids (the control).
  - Parts 2 and the fixes: every value identical to part 1's run.
  - b5b2-crevsv's five arms (what an S handler does between restarts): the pre-registered paths, two stops each,
    resumes none 2 / derive 0 / kill 1 / srev 0 / straw 0, no miss.
  - Stop latency: 4,042 cycles from an interrupt during a resumed chunk to the PARTIAL.
  - Instrument misses (my pre-registered relations, values unaffected, not widened after the fact):
    - crevint's timer-trap counts (7 against at most 4, 2 against 1): extra ticks taken at the re-fetched CREVOKE;
    - crevsv's checksum against crevint's: the page addresses move with the file's data;
    - crevsv's S traps against timer traps at the site.
- **Lint on bf8fc0cfb**: the Sublet gate passes, identical by name to 904cd9c6b's (LATCH 44, UNOPTFLAT 39,
  UNUSEDSIGNAL 831, 39 loop names). The ids55 build adds exactly two unused names, the new ex_stage ports.
- **claim-auditor** over e3ba8aed0 and 710998447: sound on the replay mark, the predicate, the saved position inside
  the contract, CRENEW and the header; one corrupt-table defect, fixed at ca1ea676d. Residuals it named, unchanged:
  - an older committed store into the table may not have drained before a lifetime op reads it (pre-existing);
  - ldc_lsu_ex outranks the replay mark at CAP_WB (pre-existing; LDC/STC bypass the DYN unit here).

## The synthesis gate on bf8fc0cfb: pre-registered (2026-11-25, before the request)

GATE KEYS (base: step 9's gate on 904cd9c6b, dcp sha16 1eebdc9602d3f0bf):
| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,954.8 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 122,911 + 2,500 = 125,411 |
Reported, not graded (predicted before the run):
- the lifetime unit (rev_node, capstone_sublet_life): +600 to +1,500 LUT and +450 to +700 FF. The new state is about
  575 flops: t, the chunk cursor, the back-link expectation, p and the release cursor (31 bits each), the chunk count
  (16), p's word 1 (64), the interrupt flop, and the saved position {v, 63-bit id, two 57-bit keys, t, stop, 32-bit
  generation};
- ex_stage: under +200 LUT (the replay mark, the predicate's flop, the straw dispatch OR);
- cva6 top: under +50 LUT (the predicate);
- core FF: +450 to +750;
- every other module: within the per-module floor.
If key 1 or 2 fails: the partition / loop members by stem, and no full build.

FULL BUILD on a pass (base: 904cd9c6b's seal, sha256 36e70855...1b08):
| key | prediction | failure branch |
|---|---|---|
| placed LUT | 148,038 + 1,000 +- 3,000 | outside: per-module placed rows against the seal |
| placed FF | 68,231 + 600 +- 2,000 | the same |
| routed | 0 errors, 0 unrouted, 0 partial | routing failure: placed area, congestion windows by module; NO WNS |
| congestion | <= 5, placed and router | above: the windows' modules |
| loops | check_timing 0, LUTLP-1 0, TIMING-23 0, each toy firing | any: members by stem |
| WNS clk_out1 | inside 904cd9c6b's null band [-1.47, +6.01] (+2.266 +- 3.74) | bad side: worst-path family and depth before any board use; good side: reported |
Reported: design WNS/TNS/WHS, slices, RAMB36 (115), worst-20 family, the count of rev_node cells on the worst 500 (POS
control: a known-cell query), sha256.

Flash: none until the seal and its kit. The silicon arms (a bare kit of b5b2-crevint / crevsv with the stop latency
pre-registered) follow from me before any flash. Launch is under your synth-guard rules.
