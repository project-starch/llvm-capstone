# Record-access trace v3: D$ residency of LB probes (nested lane, 2026-11-26/27)

These are the instruments behind the mcf and gcc v3 traces in `capstone/docs/ref/nested-allocator-workloads.md`, in the
section "mcf's record-cache population depends on the emulator's speed".

- QEMU is capstone-qemu `nested/record-trace-dc` `c3f93ba340`. It holds the tracer, the D$ model
  `target/riscv/cap_rev_dc.h` (b42c8c644's wt_dcache: 256 sets × 8 ways × 16 B, an invalid way first, else an LFSR),
  and the 16-byte capability hooks. The binary used was `d6e8ed5e`.
- `dcplugin.c` is the TCG plugin. It owns the model, feeds it every guest data read, and writes `snap=FILE` (the model's
  totals at each store to UART0's THR, physical 0x10000000). Built against QEMU's `qemu-plugin.h` and `cap_rev_dc.h`;
  the plugin used was `.so` `0136be18`.
- `dc-window.py BOOTLOG SNAP DCPLUGIN` maps the console's `@@SPEC-BEGIN/END` markers to model counts. It exits 1 when it
  finds no window or when D > 64.
- `rcm-lb-dc.c` is the rtl lane's `rcm-lb.c` (dev `6d87efe8d50a`), with an LB0 fixed at 64x4 and residency tallies per
  population. On a v2 trace it reproduces `rcm-lb`'s rows exactly.
- `rcm-lb-dc-thin.c` (2026-11-27) is `rcm-lb-dc.c` plus `--thin K:PH`: it keeps timer tick t (1-based, counted at each
  E op 0 with cause 0x80000007) only when t % K == PH, and for a dropped tick clears the flush bit on every E from the
  MTI through the first MRET after the kernel's cause-9 ecall. Without `--thin` and at `--thin 1:0` it is
  `rcm-lb-dc`. It checks the rtl lane's thinned readings (the correction to the -icount Result).
- The ticks pair (2026-11-27). The tracer at capstone-qemu `nested/record-trace-dc` `a1cbef0ede` (binary `f1d9a277`)
  writes `<trace>.ticks` under `CAPSTONE_RECTRACE_TICKS=1`: one line per MTI, with the event index, the exact
  instruction counts and the D$ model's totals.
  - `rcm-lb-dc-ticks.c` is `rcm-lb-dc-thin.c` plus `--ticks FILE`, which writes one cumulative replay row per MTI.
  - `tick-bins.py` joins the two on the event index and bins the inter-tick intervals. It also computes the
    complement test: the slice is the top LB0-miss decile and the rest is everything else.
  - `trace-spec-dc-tk.sh` and `chain-tk.sh` run mcf at shift 6 with the side file.
  - `dcstar.py` (report-only, added after the mcf-tk6 pre-registration) reads `tick-bins.py`'s complement lines and
    prints dc\*, the D$ cost at which Y/X crosses the threshold, with the MPKI ratio rest/slice. `results-tk6/` holds
    the run's `bins.out` and `dcstar.out`.
- The drivers:
  - `trace-spec-dc.sh` with `chain-dc.sh` runs the traces without icount;
  - `trace-spec-dc-ic.sh`, `boot-plugin-ic.py`, `ovl-ic/init` and `chain-ic.sh` run them under `-icount`;
  - `diag-boot.sh` and `run-b3-mcf.sh` run the B3 window check.

The committed copies replace the account's paths with `<hdd>/` and `<home>/`, so they do not run as committed.
`RAN-SHA256.txt` lists the sha256 prefixes of the files as they ran.
