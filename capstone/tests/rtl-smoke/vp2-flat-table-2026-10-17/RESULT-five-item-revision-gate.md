# The five-item module revision: QEMU gate, frame tests and the low-capacity churn control

Lane `module-resume-rd-x0`, commits c58e2f068821 (collect stopgap), ee6a1b3a13f4 (resume sfence), 250d09458739 (satp
refresh of +928), 76cfc60b3626 (busy-frame refusal), on e5775346cc8e. Module source a8643d81d12971e6.
QEMU-kit module a5dec6a32c49b3d3; board module 39d8492d5c18216b (vermagic `6.4.14 SMP riscv`, 0 compressed of 4,535).
QEMU b290f5656600 with svadu=false. Readings are result lines; logs stay outside the repo.

## Gate

| check | reading |
|---|---|
| Tier 1 (run.py) | 26/41, the same 15 FALSE and 22 exits in order as the stopgap's; console banner 2, refused 0, `collect skipped` 2 |
| chain + mpt | 16/16 identical to gate6-t3-new (compare-gate.py, banner-gated refusals 0) |
| counters (resume_sfences, resume_satp_refreshes, resume_satp_changed) | baseline (0,0,0); a step with no context (0,0,0); contract `ok` (426,546,0), and 426 = that step's CAPSTONE_VM_STATS faults= |

## Frame tests (S-21), one boot each

The child-entry tests need two supervisor slots, so they ran on kit qkit7-rev2s2: the same module, with the QEMU wrapper
appending `x-capstone-supervisor-slots=2`. Every boot's applied.log shows it, and every banner reads
`supervisor_slots=2`. On the default one-slot QEMU with supervisor_slots=2 these tests halt the guest, which is Q-32,
not a frame defect. `slot-match-gate.py --applied-log` on this kit: OK 2 = 2, exit 0; MISMATCH with supervisor_slots=1,
exit 1 (the control). Images: frame-contract.dom 359163cf6ccc681f (from frame-tests/frame-contract.c), overlap-test
47500ccd4068dd2a (from frame-tests/overlap-test.c). Halts 0 in all four.

| test | prediction | reading |
|---|---|---|
| twice | second create on a live child's frame -EBUSY, first child unharmed | first=1 second=-16; child_ran=1; console `already backs a thread` 1 |
| copy | the domain's read of its own copy of a granted frame faults | created=1, then cause=25 (exit signal 11) |
| overlap | a grant overlapping a live frame EINVAL; adjacent grants accepted | EINVAL; adjacent-above rc=0, adjacent-below rc=0; OVERLAP_TEST:PASS |
| multipage | a multi-page frame refused | rc=-22 |

## Low-capacity churn control (for the board's must-fire)

Kit qkit7-rev2, insmod `node_flat_records=4096`, one boot:

- `@@PARAM node_flat_records=4096` read back from sysfs before, between and after the steps.
- vm-contract `ok` still VM_EXIT 0 at that capacity (STATS node_capacity=4096, node_bytes=65536, faults=426;
  resume_sfences 426 = faults).
- sparse-churn: `capstone_vm: collect skipped: collection is not available on bf3db3175 (Q-27, S-entry); retired 1
  available 510, skips 1` (KERN_EMERG, console), then VM_CONTRACT_FAIL:recycling allocation, STATS nodes=3584,
  VM_EXIT:sparse-churn:1.
- At the default 65,536 the same exit is in the gate's Tier 1 (VM_CONTRACT_FAIL:recycling allocation,
  VM_EXIT:sparse-churn:1). A dedicated default-capacity boot is VOID: its 3600 s timeout included the QEMU-lock wait
  and expired before the guest started (empty log, rc 124).

## Must-fire lines, this revision

Natively KERN_EMERG: the load banner; `collect skipped` (budget 4); `thread create refused: frame ... already backs a
thread` (budget 4); `CSRUNV action N wrote no event record` (budget 4); the node_flat_records insmod refusal.
KERN_WARNING unless loaded with diag_emerg=1 (invisible on the board's level-1 console otherwise): the -ELOOP line,
`thread create refused, N supervisor slot(s) in use`, `child frame grant not retired`, `resolve refused at <site>`,
and the escape trace (also needs diag_escapes=1). The next revision makes every one of these KERN_EMERG with a budget.
