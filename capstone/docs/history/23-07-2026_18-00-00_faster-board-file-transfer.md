# Faster board file transfer (UART is the debug-session bottleneck)

## Problem

Domains + the controller are pushed to the board over the UART console by
base64+gzip, chunked, SHA-verified, reconstructed on-board. It is reliable but
slow. Root cause (measured 23-07): `fpga_console.run_command` types **every
character with a hardcoded 0.05s sleep** (`fpga_console.py:661`) because the board
UART RX FIFO silently drops chars on bulk writes; and the stock `send_file` does
**three command round-trips per chunk** (`printf`→`/tmp/part`, `sha256sum`, `cat`).

Cost example (run of 23-07): controller = 3192 b64 chars / 16 chunks ≈ 4 min;
each dom ≈ 1.5–2 min; a 4-dom session ≈ 10–11 min of transfer alone, and the
(constant) controller is re-sent every session.

## Tier 1 — DONE + VALIDATED ON BOARD (23-07, first try, no retries) — ~3x

`/tmp/capstone/fast_xfer.py::fast_put`, wired into `board_bisect_gpfree.py`:
- Appends each base64 chunk **directly** (`printf %s '<chunk>' >> file.gz`), one
  round-trip per chunk — no `/tmp/part`, no per-chunk sha, no `cat`.
- Verifies **once** at the end (decompress + whole-file sha).
- Types at a **reduced delay** (0.02s) with **larger chunks** (400). The final sha
  is the guard: on ANY mismatch it auto-retries the whole file at the safe
  0.05s/200 settings. Worst case = old speed + one retry; expected ~3x faster.
- base64 alphabet has no single quote, so chunks are safe in `'…'` unescaped.
- Offline self-test PASSED: chunked append reconstructs byte-identically, sha
  matches. Correctness does not depend on an unvalidated board assumption.

To upstream: move `fast_xfer.py` into `capstone/tests/rtl-smoke/fpga_driver/` and
have the repo `send_file`/`run_rtl_smoke` use it.

## Tier 2 — PROPOSED (bigger win; needs a one-time firmware rebuild)

The firmware already loads via HTTP upload (`/api/images/upload`) + gdb
`monitor load_image images/NAME <addr> bin` — a **fast** path that never touches
UART. Exploit it:

- **2a (recommended first):** bake the **constant controller** (`gpfree_ctl`) into
  the boot initramfs once (via the fw_payload recipe). Then each session transfers
  **only the tiny doms** over the now-fast UART (~30s each). Controller cost → 0,
  no /dev/mem work.
- **2b (max, zero UART payload):** reserve a top RAM hole (DTB `/memory` size or
  `mem=` bootarg), `monitor load_image images/vX.dom <hole_addr> bin` each dom in
  the boot gdb session, and give the controller a `/dev/mem` mmap reader that takes
  a physical address instead of a file path. Uploads are HTTP (fast); nothing but
  a few short commands go over UART.

Effort: 2a = one firmware rebuild + initramfs add. 2b = 2a + a DTB mem hole + a
`/dev/mem` controller variant. Both reusable across all future board sessions.

### Tier-2 feasibility (checked 23-07) — NOT worth it now

The rootfs is an **uncompressed cpio initramfs embedded in the kernel** at fixed
offset 0xA94ED0 (fw_payload 0x200000+; `__initramfs_start/end` are fixed symbols).
Adding a file grows the cpio and shifts everything after it, so it **cannot be
edited in place** — 2a needs a real kernel/initramfs rebuild (the boot-risky
buildroot recipe, `project_fpga_fw_payload_build_recipe`), which on the shared
board can silently break boot.

**Recommendation: stop at Tier-1.** Once Tier-1 landed, the controller transfers in
~30s (was ~4 min), so Tier-2's remaining upside is ~30–60s/session — not worth a
firmware rebuild, especially now that the gp-captable diagnosis has converged (few
more board sessions expected). Keep the Tier-2 design here for a future heavy
campaign; don't build it speculatively.

## Also cheap

- Batch all doms into one `tar` and transfer once (fewer round-trips) — minor.
- Skip re-sending a file whose on-board sha already matches (only helps if the
  board fs is persistent across power-cycles; likely tmpfs — verify on-board).
