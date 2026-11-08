# tickcost-2026-11-08

Does a tick-driven callback that runs longer than one tick period livelock the board kernel silently, and does cvprof's
tick cross that line once its table is full? This is the proposed mechanism for the three wedges with cvprof loaded
(exitprof, 8 Nov: gcc train c3f twice on board 3, perlbench test c3g once on board 2). The same stages ran 6 times
without cvprof and never wedged.

One kernel module, one boot, no new bitstream. `src/tickcost.c` builds with `build.sh` against the board kernel
(linux-6.4.14, build-fpga) using cvprof's recipe. The build has zero two-byte instructions, and `build.sh` refuses
otherwise. `images/SHA256SUMS` pins the module, and `build.sh` reproduces it byte for byte. `tickcost-guest.sh` runs
the four arms on the board guest. `PREREG.md` holds the mechanism, its evidence, and the predicted readings.

Not an RTL issue on the evidence so far. The wedge signature is fully accounted for by kernel software on this
platform, and no evidence of an interrupt/context-switch defect was found. The RTL facts it rests on are cited in
PREREG.md: the trap log's filter, the escape strip, `rdtime` emulated by M-mode firmware, and a context entered with
interrupts masked.
