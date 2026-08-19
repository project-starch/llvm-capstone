# Vendor patches — uncommitted submodule source, mirrored so it cannot be lost

These mirrors exist as a recovery path for submodule edits that would otherwise live in
one working tree only. **They are snapshots, not the source of truth.** If you change a
submodule, re-run `refresh.sh`.

The original reason was that submodule source was deliberately left uncommitted. That rule
was withdrawn on 2026-08-05 — submodule work SHOULD now be committed — which quietly broke
this tool twice over, and both were live on 2026-08-19:

* `refresh.sh` mirrored `git diff`, i.e. the WORKING TREE. Every mirrored tree was clean by
  then, so a run would have replaced all six mirrors with empty files and printed UPDATED
  for each. It now refuses to blank a non-empty mirror, and entries can pass a base commit
  so committed-but-unpushed work is captured too.
* A second reason the mirrors still matter: **not every submodule is pushable.**
  `caplifive-buildroot` and its nested `components/linux` both reject pushes from this
  account (403), so a commit there travels with the parent repo only as a patch here.

| file | mirrors | why it matters |
|---|---|---|
| `capstone-qemu.patch` | `capstone/capstone-qemu` | `CAPSTONE_GP_FABRICATE` / `CAPSTONE_GP_STANDIN` toggles in `op_helper.c`. **Every silicon-config QEMU run sets `CAPSTONE_GP_FABRICATE=0`** — without this patch the gp-free/gp-captable ABI cannot be tested under emulation at all. |
| `opensbi-component.patch`, `opensbi-capstone-sbi.patch` | `caplifive-buildroot/components/opensbi` | The component copy is what the **QEMU `fw_jump` actually builds from** (`build/local.mk`: `OPENSBI_OVERRIDE_SRCDIR`). The package copy below is *not*. |
| `buildroot-capstone-sbi-package.patch` | `caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi` | The inert large-RO copy edit. Kept because it records the intended shape of the C-4b change. |
| `capstone-reentry.c` | `caplifive-buildroot/package/modcapstone/userspace/` | **Untracked** — shares TWO regions, so the domain is entered twice (the share *is* the entry). The only way to exercise the glue's `__test_reentry` path, and the gate for S2. |
| `caplifive-buildroot-kernel-max-order.patch` | `caplifive-buildroot/configs/kernel.config` | `CONFIG_ARCH_FORCE_MAX_ORDER=13`. A domain region is one `__get_free_pages()` allocation, so the buddy allocator's maximum order caps the domain image size; this lifts it from 1,376,256 to 5,570,560 bytes, which is what an interpreter-scale image needs. |
| `transcapstone-linux-riscv-arch-force-max-order.patch` | `caplifive-buildroot/components/linux` | The `ARCH_FORCE_MAX_ORDER` Kconfig symbol, which riscv does not carry. Without it the config line above is silently dropped by `olddefconfig` — the value never takes effect and nothing says so. Both patches are needed; neither works alone. |
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
