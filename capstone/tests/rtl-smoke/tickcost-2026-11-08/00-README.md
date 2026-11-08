# tickcost-2026-11-08

Where is the hart when a run with cvprof loaded wedges? Three wedges came with cvprof (exitprof, 8 Nov: gcc train c3f
twice on board 3, perlbench test c3g once on board 2); the same stages ran 6 times without it and never wedged.

One kernel module, two boots, no new bitstream:
- `src/tickcost.c` builds with `build.sh` against the board kernel (linux-6.4.14, build-fpga) using cvprof's recipe.
  The module has zero two-byte instructions, and `build.sh` refuses otherwise. `images/SHA256SUMS` pins the module,
  and `build.sh` reproduces it byte for byte.
- `tickcost-guest.sh <ko> E1|E2` runs one boot's arms on the board guest.
- `tc-grade.py <board.txt> E1|E2` grades one boot by script (exit 0 as pre-registered, 1 a finding, 2 void). It reads
  stage output from the console and the heartbeat tails, and was dry-run on ten synthetic captures before the run.
- `PREREG.md` has the readings the wedges already give, the arms and the predicted readings. It opens with the
  withdrawal of this branch's first version (a tick-replay livelock, wrong by source).

Open: whether the stopping point is software inside cvprof's tick path or a core-level state. The single samples of
aperture 224 read an exception committing at M at every sample, which a plain software spin would not show.
