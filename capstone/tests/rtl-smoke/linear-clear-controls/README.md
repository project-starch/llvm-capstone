# Controls for `tests/scan-linear-clear-exposure.py`

The scan asks whether a **conformant** RTL — one that writes `cnull` to the linear source of
`cincoffset`/`scc`/`tighten`/`shrinkto`/`init` (R-21) and of `stc` (R-22) — would break code we
actually build. A scan like that answers "clean" both when the code is safe and when the scan is
broken, so it is worthless until it has been shown to produce the opposite answer.

`scan-controls.S` is that demonstration. It contains three functions:

| symbol | must be reported as | why |
|---|---|---|
| `must_report_break_B` | `BREAK rule-B` | `a2` is minted LINEAR by `split`, stored with `stc`, then **read** by `sd` before redefinition |
| `must_report_break_A` | `BREAK rule-A` | same `a2`, used as `rs1` of a `cincoffset` with `rd != rs1`, then read |
| `must_stay_silent` | **nothing** | identical shapes, but `a2` is `delin`'d first, so it is NONLIN and no clear is required |

The third row is the half that is easy to skip and is the more informative one: a scan that
reports everything is as useless as one that reports nothing, and only `must_stay_silent`
distinguishes the two.

## Run it

```bash
source capstone/tests/capstone-test-env.sh
llvm/cmake-build-debug/bin/clang --target=capstone64-unknown-elf -c -nostdlib -ffreestanding \
    capstone/tests/rtl-smoke/linear-clear-controls/scan-controls.S -o /tmp/scan-controls.o
python3 capstone/tests/scan-linear-clear-exposure.py /tmp/scan-controls.o
```

Expected: exactly two hits, both `BREAK`, one `rule-A` and one `rule-B`, and no mention of
`must_stay_silent`. **Anything else means the scan carries no verdict** and any corpus result
taken with it must be discarded.

## The one that has already caught a dead instrument

When the scan was first retargeted from a conservative "not provably NONLIN" rule onto
provenance, these controls went silent. The cause: `llvm-objdump` has no encoding for `SPLIT`,
`MREV` or `INIT` and prints `<unknown>`, so every linear-minting instruction was invisible and
the scan reported a clean zero having examined nothing. The scan now decodes those raw bytes
itself (`RAW_R_OPS`). Without the controls that would have shipped as "no exposure found".

## Known limit, and why the corpus is scanned in two modes

Provenance is tracked **within one symbol only**. `start-fpga-nogp.S` carves a LINEAR `a2` in
`__test_entry` and stores it in `test`, so in the default `--classes linear` mode that site is
invisible. Use `--classes linear,unknown` for the conservative sweep; it reports every operand
arriving from outside a function except `sp`/`gp`/`tp`, which the entry glue `delin`s.

`nogp.o`, built from `capstone/tests/rtl-smoke/start-fpga-nogp.S`, is the control for **that**
mode and must report `stc a2, 0x20(sp)`:

```bash
llvm/cmake-build-debug/bin/clang --target=capstone64-unknown-elf -c -nostdlib -ffreestanding \
    capstone/tests/rtl-smoke/start-fpga-nogp.S -o /tmp/nogp.o
python3 capstone/tests/scan-linear-clear-exposure.py --classes linear,unknown /tmp/nogp.o
```
