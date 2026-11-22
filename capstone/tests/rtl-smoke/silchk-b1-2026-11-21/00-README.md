# silchk-b1-2026-11-21: silicon check list for bitstream 1 (R-60, R-68, R-72, R-32)

Bare M-mode images (load at 0x80000000) whose readings close four registry entries whose fixes are in P1 but were only
ever accepted in simulation. `PREREG.md` holds the vectors and the grading; `build.sh` builds every image and checks
that the reused R-32 image reproduces byte for byte. Sources are in `tests/`, with origins in `build.sh`. One kit edit:
`niss-superpage.S` ends through CAP_PASS, so silicon prints its vector.
