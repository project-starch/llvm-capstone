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

## hold 100000 as bitstream 2's silicon discriminator: the QEMU references (pre-registered 2026-10-23, before the runs)

The board lane will run `hold.dom hold 100000` on board 2 after bitstream 2's flash, as the cheapest silicon check of
the 18-bit index. These are its QEMU references.
- **The image:** `hold.dom` 433f9f1cdcd0db9f. `capstone-cc -O2` from t3-freeze-nb-nomovc-cw-r60/sdk on dev's hold.c
  rebuilds it byte for byte.
- **The runs:** one call per boot. Neither kit sets module parameters, so both run the module's default capacity with
  no `node_flat_records`.
  - qkit7-batch2: the batch kit, the 16-bit flat table, module a05c198b;
  - qkit7-b2twin3-q8: bitstream 2's twin, 18-bit, module 33254735, launcher fad465ce.
- **Predicted:**
  - **qkit7-batch2:** `HOLD-NULL at <i>` and exit 71, no fault, high_water 64,001. hold allocates through malloc, and
    the malloc path refuses at 511 left (64,001, as mruby's smoke read), returning NULL rather than faulting. So not
    cause 30 at 64,256: that is the pymalloc pool path's stop, and hold does not take it. `<i>` is 64,001 minus the
    records held before the loop.
  - **qkit7-b2twin3-q8:** `HOLD-PEAK 100000`, `HOLD-OK 100000` and exit 0. high_water is inside 105,269..105,869, the
    paged kit's 105,569 ± 300; allocated_total is about 105,569.
