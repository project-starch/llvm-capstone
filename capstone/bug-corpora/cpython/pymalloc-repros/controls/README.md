# Controls for the virtual arms

Not cases. A program in the corpus's own format, built by `shared/build-cases.sh` with
`PYC_CASES=<this directory>`, run before the cases with the same fixture shape (event id 0).
What it must do is the configuration's, in `tools/arms.json`:

| control | virtual-pymalloc | virtual-pymalloc-pools |
|---|---|---|
| `00_control_uaf_block` -- a pymalloc block freed and read through `read_probe` | complete | fault in `read_probe` |
