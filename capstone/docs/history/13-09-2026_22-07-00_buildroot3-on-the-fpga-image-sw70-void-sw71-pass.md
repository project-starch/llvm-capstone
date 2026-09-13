# Landing buildroot #2/#3 on the FPGA image: boot sw70 VOID, boot sw71 PASS, and what the void taught (2026-09-13)

**Result.** The external collaborator's caplifive-buildroot PRs #2 (the loader reads a domain's
declared requirement) and #3 (the module sizes the domain's one region from it, order ceiling
respected) are on the FPGA image and proven on silicon by boot sw71 (22:07): control
`RESULT k800 retval=4` with the rebuilt controller, then the declaring SQLite image `f795151f3ed4883b`
at `--size 1`: `Domain requirement = 1281952 (stack 1048576)`, `SQ: A/dom-ok`, three shares,
`G/enter`, `H/return`, `Verification Hash: 111130 1e792c9d`, SPEEDTEST1-CYCLES 2,669,560,340,
HEAP 134217728. Submodule `caplifive-system/sw/buildroot` fast-forwarded to `d04bd83`, module rebaked
(`.ko ed807a292aaad3f6`), gitlinks bumped.

**Why a boot was lost first (sw70, VOID).** The ioctl struct `ioctl_dom_create_args` grew by two
fields in #2/#3, and its size is part of the `_IOWR` command number, so every host that creates
domains has to be rebuilt with the new struct or the module answers `Unrecognised IOCTL command`
(and, by its `default:` case, returns 0). Hosts linking `libcapstone.c` are fixed by a rebuild.
The board's control host in the `lpc` slot is NOT one of those: it is `ladder_perf_ctl`, a
freestanding controller (no libc, soft-float, `build-ladder-fpga.sh`) carrying a PRIVATE copy of
the struct. Grepping for the slot name found `build-ladder-base-fpga.sh` instead, which builds
`ladder_base_ctl` — the NATIVE baseline controller, with no `/dev/capstone` in it at all. Its real
`-fno-common` link failure was fixed (`static` on two file-scope objects, e78fc6ab) and its output
was staged into the `lpc` slot: same slot name, different program, six times the size. The board
reset under it and printed no RESULT. The driver's manifest check could not catch this — it pinned
the hash of the binary just built, so it proved only that the staged file was the built file.

**The two fixes and the gate.** (1) `ladder_perf_ctl.c`'s private struct grown by the same two
zeroed fields (f308efe2); the pre-edit build reproduced the board's previous `lpc` bit for bit
(2d046b38a60eea0d), so the pair is exact; on QEMU against the #3 module the old controller prints
`Unrecognised IOCTL command 3225466880` / `create_dom failed` and the new one `RESULT k800 retval=4`.
(2) A same-program gate in the board driver before staging: the builder is found by the OUTPUT
filename the driver invokes, the marker strings the driver keys on (`/dev/capstone`, `RESULT`,
`retval=`) must be present in the new binary as in the old, and a size change beyond 2x is a
different program until explained. sw71 ran under that gate.

**The rest of the private-struct set (B2, 2026-09-14).** Eight more hosts carry the same private
copy — `borrow_cost_fpga_ctl.c`, `borrow_cost_fpga_nogp_ctl.c`, `borrow_breakdown_fpga_nogp_ctl.c`,
`gpfree_fpga_ctl.c`, `rev_transferred_probe_ctl.c`, and the R01/R02/R16 repro copies of
`ladder_perf_ctl.c`. All eight pre-set `dom_id = -1` before the ioctl, so against the #3 module they
fail loudly rather than proceed with a phantom domain (verified on QEMU with the R01 copy: the
pre-grow binary, bit-identical to the prebuilt `images/ladder_perf_ctl`, `create_dom failed`; the
grown one runs `rawhazard5.dom` to `RESULT rawhazard5 retval=48879`, slots `5 5 5 5 5`). All eight
grown in one commit with the identical block. Their binaries are cited instruments, so each hash
changes at its next build and is to be noted beside the measurement row.

**Lesson recorded** (memory: stage the same program, not the same name): a slot name is not an
identity; find the builder by the output filename, compare marker strings, treat a size jump as a
different program, and prove a rebuilt host as a QEMU pair before it goes near the board.
