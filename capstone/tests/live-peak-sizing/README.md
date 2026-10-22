# Live-record peaks of the capacity cells (2026-10-22)

The evidence behind section (a) of `docs/plans/2026-10-22-next-bitstream-whole-corpus-coverage.md`.

- `hold.c`: the control program (hold N objects live; malloc/free N times). Built with the r60 cw SDK's `capstone-cc -O2`; image 433f9f1cdcd0db9f.
- `lp-parse.py`: brackets each table instance's peak and lifetime count, from the counter lines that capstone-qemu
  `measure/live-peak` prints.
- `check-control.py`: the control's verdict, against the bands pre-registered in that branch's ea7bcd757e.
- `control.txt`: the control boot's checks (QEMU bb96f69552135b93, module ed8565728d0fdabd); every one holds.
- `flat-discriminator.txt`: the same counters on the FLAT batch kit, cells 02 and 13 (pre-registered at dfc80198307c).
- `summarize.py` and `cells.tsv`: one row per call of the eleven cell boots. The module's exact allocation count lies inside
  the host bracket in every row (`in_bracket` True).

The first control boot (binary 34fe3475, ea7bcd757e) was void: it printed no closing line, because the staged runner
stops reading QEMU at the gate's end marker. Its gate refused it, and no cell ran. 730a8b2287 adds the PEAK and MINT
lines that the brackets are built from.
