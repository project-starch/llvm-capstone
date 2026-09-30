# Containerized build environment

Builds the Capstone toolchain, QEMU and the guest image inside podman, against the
**host** source tree. The image carries dependencies only — no project source is copied
into it, so every artifact lands in this working copy and survives the container.

## Quick start

```bash
capstone/container/build-image.sh                           # once
capstone/container/fetch-submodules.sh                      # on the HOST (see below)
capstone/container/run.sh capstone/container/setup.sh       # llvm + qemu + buildroot
capstone/container/run.sh capstone/container/verify.sh      # 8 checks, ends in a booted guest
capstone/container/run.sh                                   # interactive shell, env sourced
```

`setup.sh` takes stage names, and every stage is idempotent — an interrupted run is
resumed by reissuing the same command:

```bash
capstone/container/run.sh capstone/container/setup.sh qemu
capstone/container/run.sh capstone/container/setup.sh llvm buildroot
FORCE_CONFIGURE=1 capstone/container/run.sh capstone/container/setup.sh llvm
```

## Why one mount is enough

`capstone-qemu` and `caplifive-buildroot` are **submodules of this repo**, and
`capstone/tests/capstone-test-env.sh` derives every path from `CAPSTONE_REPO_ROOT`. Mount
the repo at `/work/llvm-capstone` and every default resolves with no overrides:

| host | container |
|---|---|
| `llvm-capstone/` | `/work/llvm-capstone` |
| `capstone/container/home/` | `/home/builder` (`$HOME`) |
| `capstone/container/tmp/` | `/tmp/capstone` (`$CAPSTONE_TMP_ROOT`) |

`home/` and `tmp/` are host directories on purpose: the QEMU lock, the ccache, the
Buildroot download cache and every test artifact persist between `podman run` invocations.
Both are gitignored.

## Why submodules are fetched on the host

`project-starch/capstone-qemu` is private, and this machine authenticates with
`credential.helper = store` — a GitHub token in `~/.git-credentials`. Mounting that into a
container that then compiles and runs an entire Buildroot tree hands a credential to a lot
of third-party code. Fetching source is not building, so `fetch-submodules.sh` runs
outside. If you want it inside anyway, `CAPSTONE_GIT_CREDS=1 capstone/container/run.sh ...`
bind-mounts the file read-only.

## Notes that cost time to find

- **Buildroot is driven by `caplifive-buildroot/Makefile`, never `make -C buildroot` directly.**
  The tree's Makefile is not a convenience wrapper. It regenerates the OpenSBI monitor assembly
  by running capstone-c, it passes `-DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE` (which select
  the QEMU variant of the monitor), and it keeps output in `build-qemu/` rather than a shared
  `build/`. Invoking buildroot by hand silently omits all three. Correct form:

      make TARGET=qemu CAPSTONE_CC_PATH=<repo>/capstone/capstone-c JOBS=$(nproc) setup
      make TARGET=qemu CAPSTONE_CC_PATH=<repo>/capstone/capstone-c JOBS=$(nproc) build

  `JOBS` defaults to **90** — the tree is written for a 112-core host and its own comment says a
  full-width link storm has taken that machine down. Always pass this box's count.

- **`build/` is a symlink to `build-qemu/`, created per checkout.** The Makefile deliberately
  does not create it: which target a checkout serves is a property of the checkout. Everything
  downstream (`capstone-test-env.sh`, `capstone/utils/run-qemu.sh`) reads `build/images/`.
  Two separate output directories are the fix for ISSUES.md C-11, where a shared `build/` let
  an FPGA `fw_jump.o` with an embedded device tree get relinked into the QEMU firmware, which
  then discarded QEMU's DTB and hung with zero serial output.

- **`capstone-c` is REQUIRED, despite touching nothing in capstone-test-env.sh.**
  `components/opensbi/lib/sbi/sbi_capstone_init.S:85` does `#include "sbi_capstone_dom.c.S"`,
  and those `.c.S` files are gitignored generated output. Without capstone-c the Buildroot run
  dies at OpenSBI — about an hour in, after the whole cross toolchain has been compiled.

- **`components/opensbi` has its own nested submodule**, `lib/sbi/capstone-sbi`. Missing it
  fails the same way and just as late (`fatal error: capstone-sbi/sbi_capstone.h`).

- **Editing the monitor source does not always re-sync it into the build.** Buildroot gates the
  OVERRIDE_SRCDIR rsync on `.stamp_rsynced`, so a source change after a build is silently
  ignored. Force it with `make ... opensbi-dirclean` and rebuild.

- **`python3-pexpect` is load-bearing.** It is the only non-stdlib import in
  `capstone/tests/runtime-qemu/run-domain-smoke.py`, and every runtime test funnels through
  that harness — including capstone-qemu's own revoke and mrev-codegen probe runners. Without
  it they report `no boot/fault after 3 attempts`, which reads like a guest fault rather than a
  missing host package.

- **run-smoke.sh's success markers are in the SERIAL LOG, not on stdout.** stdout carries only
  `QEMU smoke passed.`; the guest console goes to `$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log`.
  Grepping stdout for `retval = 42` reports a failure on a run that passed.

- **A stale toolchain WARNING from capstone-test-env.sh is normal here.** `setup.sh` builds a
  named subset of ninja targets, so `toolchain-fresh.py` always still sees queued work for the
  ones we never ask for. It is a warning on stderr, not a failure; sourcing still returns 0.

- **`makeinfo`/`msgfmt` warnings during Buildroot are benign** — info documentation and glibc
  message catalogs, neither of which ends up in the image. Buildroot's own host-dependency
  gate (`support/dependencies/dependencies.sh`) is the authoritative check and it passes.

### Container mechanics

- **`llvm/cmake-build-debug` is a RelWithDebInfo build.** The *name* is the contract —
  it is what `CAPSTONE_LLVM_BUILD_DIR` defaults to in `capstone/tests/capstone-test-env.sh`
  and what `build-toolchain.sh` and `toolchain-fresh.py` fall back to. Renaming it means an
  override in every shell. Assertions are on, so backend asserts still fire.

- **`capstone/tests/build-toolchain.sh` does not work in here.** It needs `systemd-run
  --user` and `$HOME/bin/logs/machine-memory.lock`, and its `-j90` / `MEMMAX=64G` defaults
  belong to a much larger host. `setup.sh` calls plain `ninja` instead. The script is left
  untouched — it is correct for the machine it was written for.

- **`git submodule update --init --recursive` fails on caplifive-buildroot.** The pinned
  commit `d04bd83b` records a gitlink at `components/opensbi-qemu` with no stanza in
  `.gitmodules`, so git has no URL and aborts the recursion. It is orphaned, not skipped at
  our peril: `configs/qemu_capstone_defconfig` selects `local-qemu.mk`, which points
  `OPENSBI_OVERRIDE_SRCDIR` at `components/opensbi`. `fetch-submodules.sh` therefore inits
  the four that are actually needed by name.

- **capstone-qemu is initialized NON-recursively.** Its 15 nested `roms/*` submodules are
  ~1 GB and a `riscv64-softmmu` build uses none of them.

- **`--userns=keep-id` is load-bearing twice over:** build outputs come back owned by you
  rather than a subuid, and Buildroot refuses to run as root.

- **`-e HOME=/home/builder` is not cosmetic.** With `keep-id` the image has no passwd entry
  for uid 1000, so `HOME` would be unset, `$CAPSTONE_QEMU_LOCK` would resolve to
  `/.capstone-locks/qemu.lock`, and its `mkdir` fails at *source* time — killing every
  script that sources the env before it does anything.

## Files

| | |
|---|---|
| `Containerfile` | ubuntu:22.04 + the LLVM, QEMU, Buildroot and Rust dependency sets |
| `pod.sh` | podman wrapper pinning `XDG_DATA_HOME` (VSCode-snap libpod mismatch) |
| `build-image.sh` | builds the image, then prints the toolchain it got |
| `fetch-submodules.sh` | **host-side**; populates capstone-qemu and caplifive-buildroot |
| `run.sh` | bind mounts + flags; no args gives an interactive shell with the env sourced |
| `setup.sh` | staged bring-up: `submodules llvm qemu buildroot` |
| `verify.sh` | eight checks, from the image toolchain through to a booted guest |
