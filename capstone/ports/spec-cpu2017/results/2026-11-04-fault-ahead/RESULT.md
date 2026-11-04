# Adaptive fault-ahead on silicon: result (board 1, bitstream 3)

Pre-registered in PREREG.md and Addenda 1-6. The kits are bkit18-fa-b3-b1 (ON: module 6e854c46c654c843 with
`fault_ahead_max=64`) and bkit18-faoff-b3-b1 (OFF, imagick only). The board ran fa-b1.sh from 4 Nov 16:55 to
5 Nov 02:51. The reference is the bitstream-3 baseline in spec/board/out. **That baseline ran on 28-29 Oct**, not on
3 Nov as the pre-registration says. Its xz and mcf stages ran on **board 2** (bkit12-perf-b3-b2), the others on
board 1.

## Grade (`fa-grade-b1.py`, rc 1)

    off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.4 vs 33.9 s F3 holds
    off 538.imagick_r-test F1 holds | faults 815 vs 815 F2 holds | real 33.1 vs 33.5 s F3 holds
    on  538.imagick_r-test F1 holds | faults 197 vs 815 F4 holds | steps 798 vs 1406 | outside 16.7 vs 17.8 s F5 rep | inside 0.9885 F6 holds | real 0.9625
    on  538.imagick_r-test F1 holds | faults 194 vs 815 F4 holds | steps 794 vs 1407 | outside 16.6 vs 17.7 s F5 rep | inside 0.9872 F6 holds | real 0.9609
    on  502.gcc_r-test-cap F1 holds | faults 1061 vs 1746 F4 holds | steps 2027 vs 2711 | outside 45.4 vs 47.1 s F5 rep | inside 0.9633 F6 FAILS | real 0.9640
    on  544.nab_r-test:spa F1 holds | faults 342 vs 447 F4 holds | steps 12617 vs 12714 | outside 192.8 vs 192.1 s F5 rep | inside 1.0008 F6 holds | real 1.0013
    on  544.nab_r-test:lif F1 holds | faults 339 vs 447 F4 holds | steps 12826 vs 12916 | outside 198.4 vs 196.7 s F5 rep | inside 1.0014 F6 holds | real 1.0025
    on  519.lbm_r-test:spa F1 holds | faults 6211 vs 104719 F4 holds | steps 24001 vs 122520 | outside 515.1 vs 631.7 s F5 holds | inside 0.9836 F6 holds | real 0.9332
    on  519.lbm_r-test:lif F1 holds | faults 6211 vs 104719 F4 holds | steps 24000 vs 122516 | outside 509.6 vs 630.6 s F5 holds | inside 0.9832 F6 holds | real 0.9308
    on  557.xz_r-test-caps F1 holds | faults 102563 vs 1694529 F4 holds | steps 165794 vs 1734180 | outside 4857.4 vs 6485.2 s F5 holds | inside 0.9271 F6 FAILS | real 0.8323
    on  505.mcf_r-test-cap F1 holds | faults 42460 vs 386606 F4 FAILS | steps 192847 vs 534689 | outside 3250.2 vs 3712.5 s F5 holds | inside 0.9929 F6 holds | real 0.9660

## Readings, Temporal arm (capstone-lifetimes)

| | faults | context entries | outside (s) | inside cycles | run time |
|---|---|---|---|---|---|
| xz | 1,694,529 → 102,563 (16.5×) | 1,734,180 → 165,794 (10.5×) | 6,485 → 4,857 | −7.3 % | **−16.8 %** |
| lbm | 104,719 → 6,211 (16.9×) | 122,516 → 24,000 | 631 → 510 | −1.7 % | **−6.9 %** |
| mcf | 386,606 → 42,460 (9.1×) | 534,689 → 192,847 | 3,713 → 3,250 | −0.7 % | **−3.4 %** |
| gcc | 1,746 → 1,061 | 2,711 → 2,027 | 47.1 → 45.4 | −3.7 % | −3.6 % |
| imagick | 815 → 194 | 1,407 → 794 | 17.7 → 16.6 | −1.3 % | −2.7 % against the same-day OFF run (−3.9 % against 28 Oct) |
| nab | 447 → 339 | 12,916 → 12,826 | 197 → 198 | +0.1 % | +0.3 % |

## What this says

- **F1 holds everywhere.** Every run has rc 0, specdiff PASS, outputs equal to the twin, instret within 0.1 %, and
  the parameter read back.
- **F4 fails on mcf, narrowly.** Faults fell 9.1×, against the pre-registered 10×.
- **The board's fault count is set by the step budget, not by the window.** Board faults are about pages / 17 on xz
  and lbm, against QEMU's 64-page windows (QEMU: xz 27,799, lbm 1,646). CV_STEP's 5 ms wall-clock budget is checked
  before each 16-page pin-ahead batch, and one pin costs about 1.2 ms at 25 MHz. So every window stops after its first
  batch. `fault_ahead_max` above 16 does nothing on the board until the budget changes. The step_budget_ns
  parameter is tested in results/2026-11-05-outside-levers.
- **F6 was mis-registered.** The in-context window includes each entry's resume: the switcher walk, the satp write's
  TLB flush, and the refill. capstone_vm.c's bench_entries comment has said so since 26 Oct. Inside cycles saved per
  removed entry are about 6,150-6,600 on lbm and xz, 7,600-8,300 on imagick and 11,000 on gcc. That predicts every
  stage, including nab's zero. The "inside cycles" column here therefore measures removed resume overhead, not
  program work.
- **OFF reproduces the baseline's faults exactly on imagick** (815 = 815). Its run time reads 1.2-1.5 % faster than
  28 Oct. That is the 28 Oct run being slow on every later session, not an effect of OFF. OFF is not measured on a
  fault-heavy stage.
- **Cross-board.** xz's and mcf's references are board 2. The board offset is not measured directly; indirect bounds
  put it at about 0.5 % or less.

N = 1 per arm. Test inputs only.
