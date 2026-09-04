# Tier-2b feasibility, and a correction: coremark_matrix is NOT transfer-blocked

**Date:** 2026-07-26 · **Lane:** B · No board time used. Read-only assessment before
committing to an image rebuild.

## The correction first

The 25-07 sweep note records `coremark_matrix` as blocked at **transfer** — "its dom
wedges the shell at every tier", and that is why it has never had a capability
verdict. I repeated that, and used it to argue tier-2b would unblock it.

**The sizes refute it.** `coremark_matrix.dom` is the **smallest** domain of the
seven:

| dom | bytes | gzipped |
|---|---:|---:|
| **coremark_matrix** | **9,560** | 1,691 |
| matmult_int | 9,680 | 1,472 |
| rv8_primes | 9,688 | 1,284 |
| beebs_recursion | 9,712 | 1,417 |
| beebs_crc32 | 9,752 | 1,407 |
| beebs_prime | 9,816 | 1,653 |
| beebs_insertsort | 9,992 | 1,687 |

`beebs_insertsort` gzips to 1,687 bytes and transfers fine; `coremark_matrix` gzips
to 1,691 — four bytes larger. There is no size story here, so a bigger delivery
channel cannot be what it needs.

**More likely the causation was inverted.** Finding #2 of the same note is that
`coremark_matrix` (`-Os`) **hangs the cscall** even as the first domain of a clean
boot. A hung domain wedges the console; the *next* transfer then fails, and the
failure was attributed to the transfer. The hang is the cause, not the consequence.

⇒ **Tier-2b is not the coremark unblock.** Test the existing dom directly instead:
it is the smallest one, so if delivery were ever going to work it works for this.
Either it runs — and we gain the 5th benchmark of the minimal paper set for one
boot — or it hangs, which *is* the missing verdict on Finding #2, on a fresh dom.

This also restores what `ref/HOW-TO-LAUNCH-ON-FPGA.md` already said and I overrode:
tier-2b is "the on-ramp for domains too big for Tier-1 transfer (SQLite-scale),
**not** a speedup for a handful of tiny integer rungs."

## What tier-2b would still legitimately buy

- **SQLite (~1 MB), goal 2.** Genuinely beyond base64-over-UART. This is real and
  remains the reason to do it.
- Removing the transfer flake that cost a boot on 26-07 — real but modest, and the
  boot+transfer retry now in `run_ladder_base_fpga.py` already mitigates it.

**It does NOT speed up the capability sweep.** The 2.5 min/rung power-cycle is forced
by the same-VA multi-domain icache hang, not by delivery. The domain-boundary
`fence.i` patch is what would address that, and it is a separate, still-untested
piece of work (task #2 remainder).

## Feasibility: how the FPGA image is actually built

Established by inspection (the in-repo `caplifive-buildroot` is **not** the tree that
produces the board image):

- The board image comes from **`caplifive-system/sw/buildroot`**, a separate
  buildroot with `BR2_TARGET_ROOTFS_INITRAMFS=y` + `BR2_TARGET_ROOTFS_CPIO_GZIP=y`.
  The in-repo `caplifive-buildroot` produces **ext2** with
  `CONFIG_INITRAMFS_SOURCE=""` — that is the QEMU rootfs, not the board's.
- The kernel embeds the rootfs: `CONFIG_INITRAMFS_SOURCE="${BR_BINARIES_DIR}/rootfs.cpio"`,
  gzip-compressed, so **the kernel must be relinked** for any rootfs change — the
  archive is not separately patchable in the image.
- Confirmed on the board: the boot log shows `Run /init as init process`, and there
  is no `root=` on the kernel command line.
- Staging dir `sw/buildroot/build/target/root/rtl-smoke/` already exists and is
  **user-owned** — that is where board binaries went before (matches
  `/root/rtl-smoke/borrow_cost_fpga.user` in the rtl-smoke README).

**Blockers to executing it here:**

| | state |
|---|---|
| `build/images/`, `build/build/linux-6.4.14/` | **root-owned** (built in a container as root) |
| `sudo` | **requires a password** — not usable non-interactively |
| `caplifive-build:latest` container | **not present locally** (`podman images` empty) — would build from scratch via `scripts/Containerfile` (base pull + toolchain + opam/OCaml for anvil) |
| `BR2_EXTERNAL_CAPSTONE_PATH` | `"/workspace/sw/buildroot"` — a **container path**; `/workspace` does not exist on the host, so a host-side `make` cannot resolve BR2_EXTERNAL without a bind-mount or symlink (needs root) |
| disk | 123 GB free (kernel tree is 1.6 GB) — not a constraint |

So the routes are: (a) rebuild the container and build inside it — the supported
path, but heavy and network-dependent; (b) sudo on the host plus a `/workspace`
symlink; (c) copy the 1.6 GB kernel tree to a user-owned dir, regenerate
`rootfs.cpio` from the user-owned `target/` with fakeroot, relink, and re-wrap with
the OpenSBI recipe already in `tests/rtl-smoke/README.md`. (c) needs no root but is
fiddly.

## Recommendation

**Postpone tier-2b; keep it scoped to SQLite (goal 2), where it is genuinely
required.** Do not carry the claim that it unblocks `coremark_matrix`.

Next board window is better spent running `coremark_matrix` as-is — smallest dom,
one boot, and it settles Finding #2 either way.
