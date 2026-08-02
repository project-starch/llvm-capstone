# target.md — rlua #19 (Row 1)

* **CVE/Issue:** rlua #19
* **Product:** rlua (Lua-in-Rust binding), crate version `0.8.2`
* **Vulnerability Type:** Heap Use-After-Free of Rust-owned memory, via Lua `__gc`
  finalizer resurrection across the FFI boundary
* **Status:** REPRODUCED (native ASan)
* **Vulnerable Commit:** `396a4b09169be429381d5db8e9bb1cd6bd5d5139`
* **Free Site:** `rlua/src/util.rs:279` (`rlua::util::destructor`), driven by Lua's
  `GCTM` / `runafewfinalizers`
* **Stale-Use Site:** `src/main.rs:35` (the `access` userdata method), reached from
  Lua via rlua's `callback_call_impl`
* **ASan Verdict:** `heap-use-after-free`, `READ of size 43`, offset 0 of the freed
  43-byte region
* **Determinism:** aborts on 10/10 consecutive runs; the free is forced at a fixed
  point by `collectgarbage("collect")`.
* **RISC-V QEMU:** not built — see `README.md`. Task spec §9 permits native-ASan-only
  artifacts where the QEMU leg is impractical.

## Toolchain note — requires a patch, which does not mask the defect

rlua at this commit calls `std::mem::uninitialized()`. Since rustc 1.48 that
aborts the process at runtime (non-unwinding panic: "attempted to leave type ...
uninitialized, which is invalid"), during `LuaTable::set` — before the trigger
ever reaches Lua's collector.

`rlua-modern-rustc.patch` replaces both call sites with `std::ptr::read`, applied
by `build.sh`. One of the two sites is the userdata destructor itself, so the
substitution was chosen to preserve the defect: `ptr::read` moves out and drops
exactly as `mem::replace(_, mem::uninitialized())` did, and likewise leaves the
slot unmarked, so a second `__gc` still double-drops. See the patch header for
the full argument.

### Leaving the destructor unpatched was tried, and it degrades the artifact

The destructor is instantiated as `destructor::<RefCell<T>>` (`rlua/src/lua.rs:1607`,
confirmed in the ASan frame below), so the rustc ≥1.48 abort fires only when `T`'s
all-uninitialised bit-pattern is invalid. `String` holds a `NonNull` and so is
invalid; a raw pointer plus a `usize` is not. Making the harness payload an
all-bits-valid heap owner therefore lets the **upstream destructor compile and run
completely unmodified** — the patch shrinks to its one off-path hunk.

It was built and run that way. It does not produce a use-after-free:

| Destructor | Result |
|---|---|
| `ptr::read` (patched) | `heap-use-after-free`, `READ of size 43`, offset 0 of the freed region, free site attributed to `destructor::<RefCell<Userdata>>` |
| `mem::uninitialized` (pristine) | `SEGV on unknown address`, "dereference of a high value address" |

The reason is that `mem::replace(obj, mem::uninitialized())` *writes* undef bytes
back into the userdata slot. The resurrected handle then reads a garbage pointer
rather than the freed one, so the fault is a wild read that ASan cannot attribute
to any allocation — a spatial-looking crash standing in for a temporal defect,
which is precisely the wrong evidence for this row.

`ptr::read` is therefore kept, and it is the more faithful reproduction: it leaves
the original bytes in place, so the stale access reads the genuinely freed buffer
and the second `__gc` double-drops the same object. The cost is that one patched
line sits on the defect path; the benefit is that the row demonstrates the temporal
defect it claims to.

This is also why the row needs **nightly** Rust: `-Zsanitizer=address` is
nightly-only.

> **Supersedes an earlier SKIPPED filing.** This row was previously recorded as
> infeasible on the grounds that modern rustc's `mem::uninitialized` ban made
> "runtime-based Use-After-Free/Double-Free verification infeasible". The ban is
> real and does abort the unpatched build, but it is a two-line toolchain
> incompatibility in rlua's internals rather than a property of the vulnerability,
> and removing it leaves the bug intact and reproducible.
