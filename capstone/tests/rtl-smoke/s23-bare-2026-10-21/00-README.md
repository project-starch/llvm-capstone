# s23-bare-2026-10-21: `s23-cpmp-rearm` as a bare M-mode JTAG image for the batch bitstream's first boot

One image, one arm, two pre-registered vectors. `PREREG.md` is the pre-registration and the grading rule; `build.sh`
rebuilds the image from `tests/` (checked equal to capstone-ariane `c721c1a2a`'s copy at build time) with the
toolchain named by `CAPSTONE_LLVM_BIN`; `images/` holds the `.bin`, its `board_rec` address (`.rec`) and `SHA256SUMS`.
`out/` (ELF) is a build product and is not committed. Cite the image by its sha256, never by its name.
