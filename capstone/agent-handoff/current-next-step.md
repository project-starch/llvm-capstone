# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is:

> prove that an `sbi.dom + .smode` payload can **reliably observe and/or mutate one host-shared region** in the current runtime path

In short: **before the full `HC_WRITE_STDOUT` RPC round trip, first prove shared-region visibility in the exact S-mode path that the split design depends on**.

## Why this is the right next step

The project already has a validated native sample-domain flow:
- in-tree `clang` compiles the sample domain,
- in-tree `ld.lld` links it as native `EM_CAPSTONE`,
- the userspace loader accepts `EM_CAPSTONE`,
- the sample domain executes successfully in the Capstone QEMU/Buildroot runtime.

So the narrowest remaining uncertainty is no longer "can we enter a Capstone domain?" It is:

> **can the current `sbi.dom + .smode` runtime path actually support the shared-memory contract that the planned host-call RPC design depends on?**

Source-backed findings from the repository point to a split model:

- userspace already has the needed domain/region primitives in
  `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c`:
  - `call_dom()`
  - `create_region()`
  - `map_region()`
  - `shared_region_annotated()`
- the host/domain boundary already supports repeated multi-round interaction patterns;
  `miniweb_frontend.c` calls a domain, does ordinary host-side Linux work, and calls it again
- the domain side already returns via `SBI_EXT_CAPSTONE_DOM_RETURN`
- the exported userspace ABI currently exposes `IOCTL_DOM_CALL`, but does **not** expose a general userspace-visible resume/trapframe ABI (`DOM_RESUME`, register snapshots, etc.)

That means the lowest-risk near-term design is still:

- **yes**: split host-enclave execution
- **yes**: shared-memory request/response ABI, but only after the basic shared-region visibility probe is proven in this path
- **yes**: ordinary Linux work performed by the host helper
- **no for now**: speculative SGX-like yield/resume ABI assumptions
- **no for now**: immediate hosted glibc/sysroot work as the main milestone

## Concrete form of the next step

1. Reuse the existing `sbi.dom` wrapper path.
2. Create exactly one shared region from a guest userspace helper.
3. Share that region into the domain using the most proven path available.
4. Run a tiny `.smode` payload that writes a sentinel value into the shared region.
5. Verify in the guest userspace helper whether the sentinel became visible.
6. Only after that works, move on to the full `HC_WRITE_STDOUT` metadata/payload protocol.

## What not to jump to yet

Do **not** jump straight to:

- full hosted `capstone64-unknown-linux-gnu` sysroot compatibility,
- `glibc` / `musl` / `picolibc` porting,
- speculative yield/resume trap ABIs,
- FFmpeg/sqlite/libpng/SPEC directly.

The point of this step is to remove the narrowest newly observed blocker in the split-runtime path, rather than assuming the full HostCall protocol before the shared-memory visibility probe is proven.

