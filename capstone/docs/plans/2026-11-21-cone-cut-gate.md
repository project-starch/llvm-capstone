# The rev-node cone cut on P1: synthesis-only gate (board lane copy of the pre-registration)

Status: pre-registered 2026-11-21, before the launch. The keys are the rtl lane's (land/rtl-cone-cut 5e3a51014bc9,
`capstone/docs/plans/2026-11-21-rev-node-cone-cut.md`), copied unchanged.

**Tree, read at source.** capstone-ariane p1-revrd-cut 5155e8e02, parent P1 d83d3a633. `git diff d83d3a633 5155e8e02`
touches `core/ex_stage.sv` only (+48/-17).
- It registers `rv_rd_res` (data, user and rvalid), with the read index and namespace flag it is decoded under.
- The grant stays combinational.
- Lint is at P1's baseline on every counter (the rtl lane's numbers).
- It cuts the crossing this lane read on P1's routed checkpoint: `rev_node/rd_data[5]` for 461 of the worst 500 paths,
  `rv_rd_res[data_rvalid]` for 39 (board/meas-pw 00cc2382f1eb, 70c6aafff17c).

**Gate.** P1's synthesis-only flow, with the same watchers, the kept checkpoint, LUTLP-1 and check_timing loops, each
with its toy control. Chain `twc/cut/post-cut.sh`, grader `twc/cut/grade-cut.py`. The grader was dry-run on B5a's gate,
where it FAILs mapping, LUT and FF as it must for that tree.

| Key | Limit |
|---|---|
| mapping | Technology Mapping peak within 10 % of P1's 5,041.7 MB; whole run inside the 40 GB guard |
| loops | 0 by every probe, each control firing |
| core LUT | P1 ± 2,500 |
| core FF | P1 + 150 to + 400 |

The full build waits for the rtl lane's pairing arms and its 95-test sweep. Its keys are in the rtl lane's file:
- WNS about -4.6 ns, inside [-8.3, -0.9]; a WNS inside P1's null band is the kill;
- a routing-failure branch;
- congestion ≤ 5.
