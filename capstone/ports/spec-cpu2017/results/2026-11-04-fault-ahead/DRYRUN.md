# fa-b2.sh dry run (4 Nov), the chain stubbed

## A-ok
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=0
ON arm chain rc=0
GRADE rc=2
restarting board 2's chain on 2 lines of $CAPSTONE_WORK/perf/fa/dryrun/A-ok/rest
DONE off 0 on 0; chain pid N
chain calls: 3
```
## A2-offclean
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r F1 holds | faults 815 vs 815 F2 holds | real 33.1 vs 33.1 s F3 holds
ON arm chain rc=0
GRADE rc=0
restarting board 2's chain on 2 lines of $CAPSTONE_WORK/perf/fa/dryrun/A2-offclean/rest
DONE off 0 on 0; chain pid N
chain calls: 3
```
## B-offfail
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=1
OFF arm did not finish cleanly: the ON arm is not run
GRADE rc=2
restarting board 2's chain on 2 lines of $CAPSTONE_WORK/perf/fa/dryrun/B-offfail/rest
DONE off 1 on skipped; chain pid N
chain calls: 2
```
## B2-offpending
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r PENDING
OFF arm not clean (control, rc, outputs, faults or time differ from the baseline): the ON arm is not run
GRADE rc=2
restarting board 2's chain on 2 lines of $CAPSTONE_WORK/perf/fa/dryrun/B2-offpending/rest
DONE off 0 on skipped; chain pid N
chain calls: 2
```
## C-notsent
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
STOP: board 2's chain ended, but not at the sentinel
x
1 CB STOP after /x/times
chain calls: 0
```
## D-prereg
```
START pid N
STOP: pre-registration deadbeef0000 is not on origin/spec-silicon
chain calls: 0
```

## S1-slot (13:40, the slot kit added)
```
START pid N
preconditions ok; waiting for board 2's chain to stop at the sentinel
chain stopped at the sentinel
SLOT slot-9 BEGIN
SLOT slot-9 END rc=0
SLOT slot-6 BEGIN
SLOT slot-6 END rc=0
SLOT slot-5 BEGIN
SLOT slot-5 END rc=0
SLOT slot-7 BEGIN
SLOT slot-7 END rc=0
SLOT slot-8 BEGIN
SLOT slot-8 END rc=0
SLOT slot-1 BEGIN
SLOT slot-1 END rc=0
SLOT slot-2 BEGIN
SLOT slot-2 END rc=0
SLOT slot-3 BEGIN
SLOT slot-3 END rc=0
SLOT slot-4 BEGIN
SLOT slot-4 END rc=0
SLOT GRADE objb rc=2
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r F1 holds | faults 815 vs 815 F2 holds | real 33.1 vs 33.1 s F3 holds
ON arm chain rc=0
GRADE rc=0
restarting board 2's chain on 2 lines of $CAPSTONE_WORK/perf/fa/dryrun/S1-slot/rest
DONE off 0 on 0; chain pid N
```

## Board 1 (fa-b1.sh, 15:50)
### A-b1ok
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.9 vs 33.9 s F3 holds
off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.5 vs 33.5 s F3 holds
ON arm chain rc=0
GRADE rc=0
launching the C.5a session (perf/c5a-first/c5a-b1.sh)
DONE off 0 on 0
calls: 3
```
### B-b1offfail
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r-test PENDING
off 538.imagick_r-test PENDING
OFF arm not clean (control, rc, outputs, faults or time differ from the baseline): the ON arm is not run
GRADE rc=2
launching the C.5a session (perf/c5a-first/c5a-b1.sh)
DONE off 0 on skipped
calls: 2
```
### C-b1notsent
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
STOP: board 1's chain ended, but not at the sentinel
x
1 CB STOP after /x/mcf-devmk
calls: 0
```
### D-b1slot (15:35)
```
START pid N
preconditions ok; waiting for board 1's chain to stop at the sentinel
chain stopped at the sentinel
SLOT GRADE bs3 rc=2
OFF arm chain rc=0
OFF arm grade: off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.9 vs 33.9 s F3 holds
off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.5 vs 33.5 s F3 holds
ON arm chain rc=0
GRADE rc=0
launching the C.5a session (perf/c5a-first/c5a-b1.sh)
DONE off 0 on 0
```
