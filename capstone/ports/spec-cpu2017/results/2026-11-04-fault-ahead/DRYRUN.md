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
