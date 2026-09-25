# tshark on dev's compiler, then without its thread-local workarounds — predictions (2026-09-25)

Written and pushed before any boot of either step.

**Why.** The tshark port and every committed tshark result were built with `b7b31421e9fa`, the
shared main-clone build. It predates the C-46 fix and C-47's thread-local support, which dev
merged in `3979abd8e9a3`. dev's compiler source has not moved since.

**How.** Two steps, each changing one thing, each in its own work directory:
- **Step 1, the compiler only.** A private build at dev's tip, in the worktree's own
  `llvm/cmake-build-debug`, configured like the main build with the revision pinned to
  `3979abd8`. The whole port is rebuilt from the same sources and patches.
- **Step 2, the thread-local workarounds only.** Patch 0004 and `deps/patches/libgcrypt-0001`
  go. The define goes from the Wireshark and GLib builds, and stays in libgcrypt's for 0002,
  which uses it as its "in a domain" selector.

The shared main-clone compiler is not touched.

**Proof the private compiler is dev's:** two TUs are preprocessed once, then compiled at the
build's own flags with `-g0`. They are tshark's `epan/except.c` and the thread-local probe's
`tls_test.c`.
- The compiler lane's build of `3979abd8` (`fixbuild`) gives the same objects on two runs:
  `except` `5e0ee897c3de0758`, `tls_test` `63d0a24e1bb67491`.
- The control fires: `b7b31421` gives a different `except` object and cannot compile `tls_test`.
- The private build must give both hashes exactly.

**Step A, already in the tree.** `host/cross-build.sh` no longer applies patch 0007, which belongs
to the sublet arm alone. Applied in the cross build it would reach every arm's libwsutil, and
then fail to apply a second time in the sublet arm. The committed results are unaffected: their
tree was built before 0007 existed.

## Step 1 predictions (the compiler only)

- **P1.1** Every dependency recipe passes its gates. The cross build completes, and `diff -r`
  of the new `xsrc` against the baseline's is empty.
- **P1.2** `build-domain.sh` passes its link gates and negative control on level0, shrink and
  sublet.
- **P1.3** Stages: M1–M5 REACHED and MATCH on level0, shrink and sublet.
- **P1.4** Oracle:
  - on level0 (2 boots): stdout MATCH on dhcp, dns_port, http, arp, their flips and dns-ooo,
    with every flip firing; ntp DIFFERS; stderr MATCH throughout;
  - on sublet the same, in 3 boots of at most four runs.
- **P1.5** Fixtures, one repeat per arm: 2 + 4 + 7 = 13 boots, every run as `safety-expect.txt`
  predicts. That file is not edited. A different outcome, most plausibly fixture 9 (merged
  globals, a codegen matter), is a finding.
- **P1.6** Heap: level0's M5 peak on dhcp is within 1 KiB of the baseline's 28,123,920. On
  sublet, one run's split + mrev is within 5% of the baseline's 12,596.

## Step 2 predictions (the thread-local workarounds only)

- **P2.1** Before any long build, three Wireshark TUs (`wiretap/wtap.c`, `wsutil/filesystem.c`,
  `epan/except.c`) and libgcrypt's `fips.c` compile at their real `-O2` without the workaround.
  Their objects reach the thread-locals tp-relatively (`R_RISCV_TPREL_*` relocations, or
  `cincoffset …, tp`).
  - Negative control: a thread-local initialised with a global's address fails with C-47's
    error.
- **P2.2** The change is contained: after `--strip-debug`, only `wtap.c.o`, `filesystem.c.o`,
  `except.c.o` and libgcrypt's `fips` object differ from step 1. GLib's stripped archive is
  identical, because its define was inert.
- **P2.3** The change is real. M5 has a non-empty PT_TLS segment, where step 1's is empty:
  - `memsz` 2,200–2,300 bytes (`errmsg_errno` ×2 at 1,025, `errbuf` 128, `stack_top` 16,
    `the_tc` 8, plus alignment);
  - `filesz` 8–32 bytes: `the_tc` has a non-zero initialiser, so `tls.c`'s template copy runs
    for the first time in any port;
  - those variables are TLS symbols.
- **P2.4** Stages, oracle and fixtures exactly as P1.3–P1.5, plus a shrink stages boot. On shrink,
  the TLS block has exact bounds with `tp` up to 4,095 bytes into it.
- **P2.5** Heap: level0's M5 `peak_end` grows by the TLS block over step 1's.
  - The block is `calloc(sizeof(struct pthread) + 4095 + memsz)`, with `sizeof(struct pthread)` =
    352 bytes on capstone64 (measured), plus level0's 48-byte header.
  - It is made first, before the constructors, so it shifts everything after it.
  - Band: +6,500 to +7,000 bytes.
- **Named risk.** On sublet, the TLS block's `calloc` is the heap's first allocation, made before
  the constructors run. It should work, because the region is shared before that entry, but the
  path has never run. If it fails, every sublet run returns status -1 with no output, and that is
  recorded as a finding.

## Results

**Step 1 (the compiler only) — landed 2026-09-25. All six predictions AS PREDICTED.**
P1.1 `diff -r` of `xsrc` empty and the cross build linked; P1.2 gates and the negative control on
all three arms; P1.3 15/15 stage cells; P1.4 the whole oracle, `ntp` differing as predicted and all
four flip controls firing; P1.5 36/36 fixture cells, fixture 9 unmoved; P1.6 level0 `peak_end`
28,123,984 (64 bytes from the baseline) and sublet `split + mrev` = 12,596, the baseline exactly.
Eight of 29 boot attempts stalled before the domain started (I-12) and passed on retry.
Full record: `ports/wireshark/app/results/2026-09-25-qemu-dev-compiler/`.

**Step 2 (the thread-local workarounds) — not yet run.** Its predictions above stand unchanged.
