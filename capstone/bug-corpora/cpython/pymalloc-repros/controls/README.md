# Controls for the virtual arms

Not cases. A program in the corpus's own format, built by `shared/build-cases.sh` with
`PYC_CASES=<this directory>`, run before the cases with the same fixture shape (event id 0).
What it must do is the configuration's, in `tools/arms.json`:

| control | virtual-pymalloc | virtual-pymalloc-pools |
|---|---|---|
| `00_control_uaf_block` -- a pymalloc block freed and read through `read_probe` | complete | fault in `read_probe` |

`nested-lifetime/` is not one of these: the parent, sibling and successor controls of
`docs/plans/2026-10-11-nested-lifetime-controls.md`, run in the whole interpreter with patch 0014
(`CPY_SUBLET=1`) against the release build. Its name keeps it out of `build-cases.sh` and
`run-virtual.py`; results in `results/2026-10-11-controls/`.
