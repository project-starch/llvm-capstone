# Result lines, 2026-10-07 (CheriBSD default: revocation on, async, 1/4)

## R1 heap per live byte (plot-scatter.py)
program        peak live MiB peak alloc MiB  ratio
glibc-simple            0.12          10.83  86.74   ./glibc-simple
cfrac                   0.92          10.73  11.62   ./cfrac 17545186520507317056371138836327483792789528
espresso                1.15          11.60  10.10   ./espresso largest.espresso
mstress                 3.61           9.57   2.65   ./mstress 1 25 25
mstress                 7.40          17.27   2.33   ./mstress 1 50 25
mstress                 7.40          17.27   2.33   ./mstress 1 50 25
mstress                11.37          29.03   2.55   ./mstress 1 100 25
mstress                29.81          68.46   2.30   ./mstress 1 200 25
mstress                64.15         150.19   2.34   ./mstress 1 400 25
alloc-test             12.54          25.36   2.02   ./alloc-test 1
sh6bench              295.15         579.33   1.96   ./sh6bench 1
malloc-large          436.00         805.11   1.85   ./malloc-large
barnes                894.04         894.13   1.00   ./barnes < input

## R4 ratio knob, alloc-test (plot-ratio.py)
setting ./alloc-test 1
     r    min    max   mean model min model max passes sweep pages/1e6
0.0625   1.10   1.16   1.13      1.07      1.14   1911           98589
0.1250   1.20   1.36   1.27      1.17      1.33    819           48055
0.2500   1.48   2.03   1.77      1.50      2.00    273           21820
0.5000   3.06  32.01  16.86  diverges               15           12733

## R5 reuse distance (plot-reuse-cdf.py)
./glibc-simple                   off  reuses=95996227 median<=128 share<=2^12=0.9998 fresh_share=0.0000
./espresso largest.espresso      on   reuses=32417004 median<=131072 share<=2^12=0.0064 fresh_share=0.0326
./glibc-simple                   on   reuses=95084676 median<=524288 share<=2^12=0.0000 fresh_share=0.0095
./mstress 1 50 25                on   reuses=168810 median<=16384 share<=2^12=0.0036 fresh_share=0.1000

## R6 where the bytes go (plot-frag.py)
program                      resident MiB asked MiB      asked   rounding quarantine      holes      dirty   (share of mean resident)
./glibc-simple                      15.02      0.08      0.006      0.000      0.450      0.001      0.543   samples=1796
./cfrac 17545186520507317056        15.32      0.62      0.041      0.005      0.416      0.176      0.363   samples=11277
./espresso largest.espresso         16.71      0.72      0.043      0.001      0.389      0.025      0.542   samples=3604
./alloc-test 1                      30.40     11.45      0.377      0.034      0.316      0.112      0.162   samples=12238
./mstress 1 200 25                  68.23     24.42      0.358      0.018      0.526      0.007      0.091   samples=89
./barnes < input                   902.45    810.04      0.898      0.093      0.000      0.000      0.009   samples=1

## R7 mapping owners (plot-maps.py)
alloc-test_1         samples=34 peak total=   37.0 MiB  jemalloc=26.7 mrs=5.9 file=3.4 other=0.9  mrs peak share=16.3%  time -l maxrss=39.2
glibc-simple         samples=58 peak total=   35.4 MiB  jemalloc=21.4 mrs=11.1 file=2.1 other=0.8  mrs peak share=41.9%  time -l maxrss=35.7
mstress_1_200_25     samples=3 peak total=   78.7 MiB  jemalloc=74.3 mrs=1.2 file=2.1 other=1.1  mrs peak share=1.7%  time -l maxrss=80.0
sh6bench             samples=98 peak total=  749.2 MiB  jemalloc=601.1 mrs=139.7 file=2.1 other=6.2  mrs peak share=26.3%  time -l maxrss=750.8

## R8 Sublet fit prediction (fit.py)
program                       peak MiB  objects  pow2-256 MiB  x held  pool 4 MiB    ids
cfrac 1754518652050731705637      0.92    24381          5.96    6.46   too small   fits
espresso largest.espresso         1.17     4408          1.89    1.62        fits   fits
glibc-simple                      0.12     1601          0.42    3.38        fits   fits
mstress 1 25 25                   4.26     2856          5.35    1.26   too small   fits
mstress 1 50 25                   7.43     5522          9.53    1.28   too small   fits
