# c5a-b1.sh dry run (4 Nov), every board call stubbed

Stubs: busy, hub, flash, readback, store-has, run_sup_bare.py, chain-board.sh and the three kit graders (exit code set per branch). Paths below are relative to $CAPSTONE_WORK.

The first E run found that a HARD STOP during the kits still let SPEC run, because the stub graders read rc 0. The script
now blocks SPEC after any HARD STOP (`hard`), and A2, E2 and E3 are the re-runs on the fixed script. B, C, D, F, G and H
ran on the version before the fix, which differs only in that SPEC gate.

## A2-success
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 BEGIN
IMAGE hpm-5 END rc=0
IMAGE hpm-3 BEGIN
IMAGE hpm-3 END rc=0
IMAGE hpm-4 BEGIN
IMAGE hpm-4 END rc=0
IMAGE hpm-6 BEGIN
IMAGE hpm-6 END rc=0
IMAGE hpm-1 BEGIN
IMAGE hpm-1 END rc=0
IMAGE hpm-2 BEGIN
IMAGE hpm-2 END rc=0
IMAGE hpm-7 BEGIN
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl BEGIN
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale BEGIN
IMAGE objb-stale END rc=0
IMAGE objb-stale-w BEGIN
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 BEGIN
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 BEGIN
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 BEGIN
IMAGE objb-perf-s3 END rc=0
IMAGE objb-perf-s4 BEGIN
IMAGE objb-perf-s4 END rc=0
IMAGE objb-perf-s5 BEGIN
IMAGE objb-perf-s5 END rc=0
IMAGE objb-perf-s6 BEGIN
IMAGE objb-perf-s6 END rc=0
IMAGE objb-perf-s7 BEGIN
IMAGE objb-perf-s7 END rc=0
IMAGE objb-pcc BEGIN
IMAGE objb-pcc END rc=0
HPM GRADE rc=0
OBJB GRADE rc=0
SPEC on C.5a BEGIN
SPEC on C.5a END chain rc=0
SPEC GRADE rc=2
  gcc      Full     PENDING
  imagick  Full     PENDING
  imagick  MapLife  PENDING
  nab      Full     PENDING
  nab      MapLife  PENDING
  lbm      Full     PENDING
  lbm      MapLife  PENDING
  xz       Full     PENDING
  mcf      Full     PENDING
  NO DATA: no finished C.5a stage under $CAPSTONE_WORK/perf/c5a-first/dryrun/A2-success/specout
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/A2-success/rest.list
DONE r67 0 hpm 0 objb 0 spec 0; chain pid N
calls: 22 image runs; 2 flashes; 2 chain calls
```
## B-r67hole
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=1
R-67 not FIXED: nothing else runs on C.5a (it goes to the lead)
SPEC on C.5a skipped (r67 1 hpm x objb x)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/B-r67hole/rest.list
DONE r67 1 hpm x objb x spec skipped; chain pid N
calls: 4 image runs; 2 flashes; 1 chain calls
```
## C-c5flashfail
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=1
C.5a flash failed
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/C-c5flashfail/rest.list
DONE r67 x hpm x objb x spec skipped; chain pid N
calls: 0
0 image runs; 2 flashes; 1 chain calls
```
## D-notsentinel
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
STOP: the chain ended, but not at the sentinel; nothing flashed
x BEGIN y
12:00:00 CB STOP after 505.mcf_r-test-capstone-lifetimes-devmk
calls: 0 image runs; 0 flashes; 0 chain calls
```
## E2-hardstop
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 BEGIN
IMAGE hpm-5 END rc=0
IMAGE hpm-3 BEGIN
IMAGE hpm-3 END rc=0
IMAGE hpm-4 BEGIN
IMAGE hpm-4 END rc=0
HARD STOP at hpm-4: resident is not caplifive_vp2c_1fc2cd5aa
HPM GRADE rc=0
OBJB GRADE rc=0
SPEC on C.5a skipped (r67 0 hpm 0 objb 0 hard-stop 1)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/E2-hardstop/rest.list
DONE r67 0 hpm 0 objb 0 spec skipped; chain pid N
calls: 7 image runs; 2 flashes; 1 chain calls
```
## E3-hardstop-objb
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 BEGIN
IMAGE hpm-5 END rc=0
IMAGE hpm-3 BEGIN
IMAGE hpm-3 END rc=0
IMAGE hpm-4 BEGIN
IMAGE hpm-4 END rc=0
IMAGE hpm-6 BEGIN
IMAGE hpm-6 END rc=0
IMAGE hpm-1 BEGIN
IMAGE hpm-1 END rc=0
IMAGE hpm-2 BEGIN
IMAGE hpm-2 END rc=0
IMAGE hpm-7 BEGIN
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl BEGIN
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale BEGIN
IMAGE objb-stale END rc=0
IMAGE objb-stale-w BEGIN
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 BEGIN
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 BEGIN
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 BEGIN
IMAGE objb-perf-s3 END rc=0
HARD STOP at objb-perf-s3: resident is not caplifive_vp2c_1fc2cd5aa
HPM GRADE rc=0
OBJB GRADE rc=0
SPEC on C.5a skipped (r67 0 hpm 0 objb 0 hard-stop 1)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/E3-hardstop-objb/rest.list
DONE r67 0 hpm 0 objb 0 spec skipped; chain pid N
calls: 17 image runs; 2 flashes; 1 chain calls
```
## F-b3flashfail
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 BEGIN
IMAGE hpm-5 END rc=0
IMAGE hpm-3 BEGIN
IMAGE hpm-3 END rc=0
IMAGE hpm-4 BEGIN
IMAGE hpm-4 END rc=0
IMAGE hpm-6 BEGIN
IMAGE hpm-6 END rc=0
IMAGE hpm-1 BEGIN
IMAGE hpm-1 END rc=0
IMAGE hpm-2 BEGIN
IMAGE hpm-2 END rc=0
IMAGE hpm-7 BEGIN
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl BEGIN
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale BEGIN
IMAGE objb-stale END rc=0
IMAGE objb-stale-w BEGIN
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 BEGIN
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 BEGIN
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 BEGIN
IMAGE objb-perf-s3 END rc=0
IMAGE objb-perf-s4 BEGIN
IMAGE objb-perf-s4 END rc=0
IMAGE objb-perf-s5 BEGIN
IMAGE objb-perf-s5 END rc=0
IMAGE objb-perf-s6 BEGIN
IMAGE objb-perf-s6 END rc=0
IMAGE objb-perf-s7 BEGIN
IMAGE objb-perf-s7 END rc=0
IMAGE objb-pcc BEGIN
IMAGE objb-pcc END rc=0
HPM GRADE rc=0
OBJB GRADE rc=0
SPEC on C.5a BEGIN
SPEC on C.5a END chain rc=0
SPEC GRADE rc=2
  gcc      Full     PENDING
  imagick  Full     PENDING
  imagick  MapLife  PENDING
  nab      Full     PENDING
  nab      MapLife  PENDING
  lbm      Full     PENDING
  lbm      MapLife  PENDING
  xz       Full     PENDING
  mcf      Full     PENDING
  NO DATA: no finished C.5a stage under $CAPSTONE_WORK/perf/c5a-first/dryrun/F-b3flashfail/specout
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=1
STOP: the flash back to bitstream 3 failed; board 1 left as is, chain NOT restarted
calls: 22 image runs; 2 flashes; 1 chain calls
```
## G-badprereg
```
START pid N
STOP: pre-registration deadbeef0000 is not on origin/spec-silicon
calls: 0 image runs; 0 flashes; 0 chain calls
```
## H-hpmfail
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke BEGIN
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged BEGIN
IMAGE t1-untagged END rc=0
IMAGE t1-oob BEGIN
IMAGE t1-oob END rc=0
IMAGE t1-fire BEGIN
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 BEGIN
IMAGE hpm-5 END rc=0
IMAGE hpm-3 BEGIN
IMAGE hpm-3 END rc=0
IMAGE hpm-4 BEGIN
IMAGE hpm-4 END rc=0
IMAGE hpm-6 BEGIN
IMAGE hpm-6 END rc=0
IMAGE hpm-1 BEGIN
IMAGE hpm-1 END rc=0
IMAGE hpm-2 BEGIN
IMAGE hpm-2 END rc=0
IMAGE hpm-7 BEGIN
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl BEGIN
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale BEGIN
IMAGE objb-stale END rc=0
IMAGE objb-stale-w BEGIN
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 BEGIN
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 BEGIN
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 BEGIN
IMAGE objb-perf-s3 END rc=0
IMAGE objb-perf-s4 BEGIN
IMAGE objb-perf-s4 END rc=0
IMAGE objb-perf-s5 BEGIN
IMAGE objb-perf-s5 END rc=0
IMAGE objb-perf-s6 BEGIN
IMAGE objb-perf-s6 END rc=0
IMAGE objb-perf-s7 BEGIN
IMAGE objb-perf-s7 END rc=0
IMAGE objb-pcc BEGIN
IMAGE objb-pcc END rc=0
HPM GRADE rc=1
OBJB GRADE rc=0
SPEC on C.5a skipped (r67 0 hpm 1 objb 0)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/H-hpmfail/rest.list
DONE r67 0 hpm 1 objb 0 spec skipped; chain pid N
calls: 22 image runs; 2 flashes; 1 chain calls
```

## Addendum: the slot kit added (13:40)
### S1-slot-success
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
SLOT bs3 slot-9 END rc=0
SLOT bs3 slot-6 END rc=0
SLOT bs3 slot-5 END rc=0
SLOT bs3 slot-7 END rc=0
SLOT bs3 slot-8 END rc=0
SLOT bs3 slot-1 END rc=0
SLOT bs3 slot-2 END rc=0
SLOT bs3 slot-3 END rc=0
SLOT bs3 slot-4 END rc=0
SLOT GRADE bs3 rc=2
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged END rc=0
IMAGE t1-oob END rc=0
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 END rc=0
IMAGE hpm-3 END rc=0
IMAGE hpm-4 END rc=0
IMAGE hpm-6 END rc=0
IMAGE hpm-1 END rc=0
IMAGE hpm-2 END rc=0
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale END rc=0
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 END rc=0
IMAGE objb-perf-s4 END rc=0
IMAGE objb-perf-s5 END rc=0
IMAGE objb-perf-s6 END rc=0
IMAGE objb-perf-s7 END rc=0
IMAGE objb-pcc END rc=0
HPM GRADE rc=0
OBJB GRADE rc=0
SLOT c5a slot-9 END rc=0
SLOT c5a slot-6 END rc=0
SLOT c5a slot-5 END rc=0
SLOT c5a slot-7 END rc=0
SLOT c5a slot-8 END rc=0
SLOT c5a slot-1 END rc=0
SLOT c5a slot-2 END rc=0
SLOT c5a slot-3 END rc=0
SLOT c5a slot-4 END rc=0
SLOT GRADE c5a rc=2
SPEC on C.5a BEGIN
SPEC on C.5a END chain rc=0
SPEC GRADE rc=2
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/S1-slot-success/rest.list
DONE r67 0 hpm 0 objb 0 spec 0; chain pid N
```
### S2-slot-r67hole
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
SLOT bs3 slot-9 END rc=0
SLOT bs3 slot-6 END rc=0
SLOT bs3 slot-5 END rc=0
SLOT bs3 slot-7 END rc=0
SLOT bs3 slot-8 END rc=0
SLOT bs3 slot-1 END rc=0
SLOT bs3 slot-2 END rc=0
SLOT bs3 slot-3 END rc=0
SLOT bs3 slot-4 END rc=0
SLOT GRADE bs3 rc=2
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged END rc=0
IMAGE t1-oob END rc=0
IMAGE t1-fire END rc=0
R67 GRADE rc=1
R-67 not FIXED: nothing else runs on C.5a (it goes to the lead)
SPEC on C.5a skipped (r67 1 hpm x objb x hard-stop 0)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/S2-slot-r67hole/rest.list
DONE r67 1 hpm x objb x spec skipped; chain pid N
```
### S3-slot-hardc5
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
SLOT bs3 slot-9 END rc=0
SLOT bs3 slot-6 END rc=0
SLOT bs3 slot-5 END rc=0
SLOT HARD STOP at slot-5: resident is not caplifive_vp2c_e7af48457
the bitstream 3 slot run stopped early
FLASH BEGIN caplifive_vp2c_1fc2cd5aa booking 999
FLASH END caplifive_vp2c_1fc2cd5aa rc=0
READBACK caplifive_vp2c_1fc2cd5aa ok rc=3
bare kits booking 999
IMAGE t1-norevoke END rc=0
IMAGE t1-untagged END rc=0
IMAGE t1-oob END rc=0
IMAGE t1-fire END rc=0
R67 GRADE rc=0
IMAGE hpm-5 END rc=0
IMAGE hpm-3 END rc=0
IMAGE hpm-4 END rc=0
IMAGE hpm-6 END rc=0
IMAGE hpm-1 END rc=0
IMAGE hpm-2 END rc=0
IMAGE hpm-7 END rc=0
IMAGE objb-pcc-ctrl END rc=0
IMAGE objb-stale END rc=0
IMAGE objb-stale-w END rc=0
IMAGE objb-perf-s1 END rc=0
IMAGE objb-perf-s2 END rc=0
IMAGE objb-perf-s3 END rc=0
IMAGE objb-perf-s4 END rc=0
IMAGE objb-perf-s5 END rc=0
IMAGE objb-perf-s6 END rc=0
IMAGE objb-perf-s7 END rc=0
IMAGE objb-pcc END rc=0
HPM GRADE rc=0
OBJB GRADE rc=0
SLOT c5a slot-9 END rc=0
SLOT c5a slot-6 END rc=0
SLOT c5a slot-5 END rc=0
SLOT HARD STOP at slot-5: resident is not caplifive_vp2c_1fc2cd5aa
SPEC on C.5a skipped (r67 0 hpm 0 objb 0 hard-stop 1)
FLASH BEGIN caplifive_vp2c_e7af48457 booking 999
FLASH END caplifive_vp2c_e7af48457 rc=0
READBACK caplifive_vp2c_e7af48457 ok rc=3
restarting board 1's chain on 2 lines of $CAPSTONE_WORK/perf/c5a-first/dryrun/S3-slot-hardc5/rest.list
DONE r67 0 hpm 0 objb 0 spec skipped; chain pid N
```
