# va-alloc-bare-2026-10-23

Item 6 (generations in virtual node ids) as a bare JTAG arm for the batch bitstream's silicon grading: `va-alloc.S`'s
VA_CSRET section built with the s23 kit's wrapper. `PREREG.md` holds the two pre-registered vectors (batch and
pre-batch) with their simulation provenance and the grading rule; `build.sh` rebuilds the image and checks the source
against the tree named by `VP_SRC`; `images/SHA256SUMS` pins the artifact.
