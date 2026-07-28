# Vendor patches — uncommitted submodule source, mirrored so it cannot be lost

The repo rule is that **submodule source stays uncommitted** (firmware/buildroot/QEMU
edits are local experiments, and the submodules have their own upstreams). That rule is
about *where the source of truth lives*. It is not a reason to have no backup: several of
these edits are load-bearing for every test run, and until 2026-07-28 the only copies of
some of them lived in `/tmp` and in a peer session's scratchpad — one cleanup away from
being unrecoverable.

So: the submodules stay dirty and uncommitted as before, and these mirrors exist purely as
a recovery path. **They are snapshots, not the source of truth.** If you change a
submodule, re-run `refresh.sh`.

| file | mirrors | why it matters |
|---|---|---|
| `capstone-qemu.patch` | `capstone/capstone-qemu` | `CAPSTONE_GP_FABRICATE` / `CAPSTONE_GP_STANDIN` toggles in `op_helper.c`. **Every silicon-config QEMU run sets `CAPSTONE_GP_FABRICATE=0`** — without this patch the gp-free/gp-captable ABI cannot be tested under emulation at all. |
| `opensbi-component.patch`, `opensbi-capstone-sbi.patch` | `caplifive-buildroot/components/opensbi` | The component copy is what the **QEMU `fw_jump` actually builds from** (`build/local.mk`: `OPENSBI_OVERRIDE_SRCDIR`). The package copy below is *not*. |
| `buildroot-capstone-sbi-package.patch` | `caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi` | The inert large-RO copy edit. Kept because it records the intended shape of the C-4b change. |
| `capstone-diag.c` | `caplifive-buildroot/package/modcapstone/userspace/` | **Untracked** — the I-3 fix. A separate domain loader so a probe can run under QEMU in seconds; deliberately not a change to `capstone-test.c`, which loads the whole QEMU corpus. Nothing but this mirror held it. |

## Not mirrored here, on purpose

The known-good monitor assembly (`sbi_capstone_dom.c.S`, md5 `b7baff6f`) and the known-good
`fw_jump.elf` (md5 `6724bcb3`) are 105 KB and 1.7 MB of generated output. They live at
`~/capstone-b-artifacts/monitor-known-good/`. They are what you restore to if a monitor
rebuild goes wrong — see `history/28-07-2026_16-10-00_monitor-regen-SOLVED-stale-fdt-object.md`.

## Refresh

    bash capstone/tests/vendor-patches/refresh.sh

Re-snapshots every patch above from the current submodule working trees and reports what
changed. Run it after touching submodule source.
