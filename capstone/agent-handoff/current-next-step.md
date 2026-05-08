# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is:

> build a **minimal synchronous host-call prototype**: a tiny Capstone domain that requests `HC_WRITE_STDOUT` through **shared metadata + shared buffer regions**, returns to the host with `DOM_RETURN`, and is invoked a second time to observe the host response

In short: **prove the split host-enclave RPC model with existing region-sharing primitives before doing libc/sysroot bring-up**.

## Why this is the right next step

The project already has a validated native sample-domain flow:
- in-tree `clang` compiles the sample domain,
- in-tree `ld.lld` links it as native `EM_CAPSTONE`,
- the userspace loader accepts `EM_CAPSTONE`,
- the sample domain executes successfully in the Capstone QEMU/Buildroot runtime.

So the narrowest remaining uncertainty is no longer "can we enter a Capstone domain?" It is:

> **what is the cheapest architecture for host services / libc-facing I/O that matches the runtime which already exists?**

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

That means the lowest-risk near-term design is:

- **yes**: split host-enclave execution
- **yes**: shared-memory request/response ABI
- **yes**: ordinary Linux work performed by the host helper
- **no for now**: speculative SGX-like yield/resume ABI assumptions
- **no for now**: immediate hosted glibc/sysroot work as the main milestone

## Concrete form of the next step

1. Define a tiny shared-memory HostCall ABI v0, for example:
   - metadata region: `phase/opcode/offset/length/result/error`
   - payload region: bytes for the request body
2. Reuse existing `create_region()/map_region()/shared_region_annotated()` infrastructure.
3. Implement a host harness that:
   - creates the domain,
   - shares metadata + payload regions,
   - calls the domain,
   - if the domain returns `HC_PENDING`, performs host-side `write(1, ...)`,
   - writes the result back into metadata,
   - calls the domain a second time.
4. Implement a tiny domain that:
   - writes `"hello from domain\n"` into the shared buffer,
   - writes a request into shared metadata,
   - returns to the host,
   - on the second invocation, reads the host result and exits cleanly.
5. Validate the full round-trip under the existing QEMU/runtime harness.

## What not to jump to yet

Do **not** jump straight to:

- full hosted `capstone64-unknown-linux-gnu` sysroot compatibility,
- `glibc` / `musl` / `picolibc` porting,
- speculative yield/resume trap ABIs,
- FFmpeg/sqlite/libpng/SPEC directly.

The point of this step is to prove the cheapest realistic host-service architecture using primitives that are already present in the repository.

