# C5 on the virtual gates: results (2026-11-25)

Pre-registered in PREREG.md (c1d8f3013f90; amendment 1 3f4e0e22e18b; amendment 2 cd97c2dced06, each pushed before its
run). The arms, the platform (kit-c5) and the image identities are recorded there.

## Verdict

`gates/compare.py arm-ctl arm-trt`: **VERDICT: IDENTICAL** (exit 0). All nine gates ran every row on both sides:

| gate | rows ran | control | treatment |
|---|---|---|---|
| hosted | 3/3 | rc=0 | rc=0 |
| runtime | 25/25 (sqlite write/read, mruby, perl exit 0) | rc=0 | rc=0 |
| nodes | 4/4 | 16 PASS | 16 PASS |
| malloc | 18/18 | 24 PASS | 24 PASS |
| pin-ahead | 21/21 | 65 PASS | 65 PASS |
| fp | 1/1 | rc=0 | rc=0 |
| pthread | 5/5 | rc=0 | rc=0 |
| step-hooks | 49/49 | 399 PASS | 399 PASS |
| memcached | 29/29 | 103 PASS, FAIL fx18_prediction, fx18_trigger, fx24_trigger | the same |

Every module load on both arms reported `SUPERVISOR_SLOTS:32`, from 1 to 30 lines per gate. Every memory-mode check
passed (`N`, requested off).

The control met amendment 2's bar before the treatment ran: rc=0 on the eight gates other than memcached, and
memcached's FAIL set exactly {fx18_prediction, fx18_trigger, fx24_trigger}.
- fx24 is the fixture the managed-mallocng record leaves unresolved.
- fx18 reads "storage NOT reissued: triggering condition not created", as in
  ports/memcached/app/results/2026-11-11-sublet-crenew, where its precondition failed on that conversion's base
  revision too. Both arms carry it, and it is not bisected here.

The comparison is not vacuous: 13 of the 17 images differ between the arms, and no libc archive does (the table in PREREG.md). The control SDK
plus the two flags reproduces the treatment's contract.dom and hosted.dom byte for byte.

## The runs that did not count

1. **Run 1** (dev 634fd5531e99, the harness as it was): 5 of 9 gates failed, so the stop rule held and the treatment
   did not run. The cause was the harness, not the kit:
   - the module's one default supervisor slot on a 32-slot CPU ("thread create refused, 1 supervisor slot in use");
   - node_count_reported, whose `NODE_COUNT_USED` line no contract ever printed.
   My pre-registered bar (rc=0 on all nine) was stricter than run-arm.sh's own convention, which compares FAIL sets.
   The semantics owner kept the stricter bar.
2. **Run 2** (harness c51a2bb381ed): eight gates passed. The slot-1 positive control fired (SLOTS pthread ['1'], one
   refusal line, pthread rc=1). Two defects remained:
   - my readback missed the runtime gate's slots line, which run.py's log filter dropped;
   - memcached failed the four _stderr checks on the launcher's own exit line.
   The treatment did not run.
3. **Run 3** (harness c51a2bb381ed + bf99f32040c6 + 8ad5c6a7b805, landed on dev as ba66310faf14): the verdict above.

A launch attempt between runs 2 and 3 ran nothing: the slot-1 control's CPU argument reached bash as a command name,
rc 127. It was stopped before any gate completed, and its partial output is kept as gates-run2-aborted.

## Platform deviation

kit-c5 boots caplifive-buildroot build-qemu's Image, firmware and rootfs (d04bd83b13cd). Dev's buildroot pin is
8fd1ea1249a9, 18 commits later. Those commits change only modcapstone and the opensbi monitor pin, both on the
physical route.
