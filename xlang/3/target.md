# target.md — GHSA-f56g-chqp-22m9 (Row 3)

* **Advisory:** GHSA-f56g-chqp-22m9
* **Product:** `libpulse-binding` (Rust binding to PulseAudio's libpulse), crate
  version **2.4.0**
* **Vulnerability Type:** Rust→C use-after-free — an iterator holds a raw copy of a
  C object pointer with no lifetime tie to the Rust owner that frees it
* **Status:** REPRODUCED (native, valgrind memcheck)
* **Vulnerable Commit:** `b6b1010847c1eb2d3a533820c8ff5cdbf9993d9e` (crate 2.4.0)
* **Fix Commit:** `9e31c82` ("proplist: fix `Iterator` use-after-free"), released in
  **2.5.0**, whose changelog is headed *"Note: This includes a security fix!"*
* **Affected range:** the changelog states the defect goes back to **1.0.5**.
* **Free Site:** `<Proplist as Drop>::drop` — `pulse-binding/src/proplist.rs:453`
  (`pa_proplist_free`), reached from
  `<Proplist as IntoIterator>::into_iter` — `proplist.rs:186`
* **Stale-Use Site:** `<proplist::Iterator as Iterator>::next` —
  `proplist.rs:171`, which calls into `pa_proplist_iterate` → `pa_hashmap_iterate`
* **Verdict:** valgrind `Invalid read of size 8`, 32 bytes inside a freed
  1,072-byte block
* **Determinism:** detected on 10/10 consecutive runs. No GC, threading, or
  allocation-layout dependence — the free and the use are adjacent statements.
* **RISC-V QEMU:** not built — see `README.md`. Task spec §9 permits native-only
  artifacts where the QEMU leg is impractical.

## Mechanism

At 2.4.0 (`pulse-binding/src/proplist.rs`):

```rust
pub struct Iterator {                  // no lifetime parameter at all
    ptr: *const ProplistInternal,      // raw copy of the C pa_proplist*
    state: *mut c_void,
}

pub fn iter(&self) -> Iterator {       // borrows &self ...
    Iterator::new(self.ptr)            // ... but the returned value is unbound
}

impl IntoIterator for Proplist {
    fn into_iter(self) -> Self::IntoIter {
        self.iter()                    // `self` is dropped when this returns
    }
}

impl Drop for Proplist {
    fn drop(&mut self) {
        if !self.weak { unsafe { capi::pa_proplist_free(self.ptr) }; }
        ...
```

`iter()` copies the raw pointer out from behind a `&self` borrow into a struct with
no lifetime, so the borrow checker stops tracking the relationship. `into_iter()`
then makes it trivially reachable: it takes `self` **by value**, copies the pointer
via `iter()`, and drops `self` on return — running `Drop` and calling
`pa_proplist_free()`. The `Iterator` it hands back is dangling before the caller
ever touches it.

Upstream's own advisory text: *"There was no actual lifetime association however
linking the lifetime of the `Iterator` object to the `Proplist` object, and thus it
was possible for the `Proplist` object to be destroyed first, leaving the
`Iterator` object working on a freed C object. This is unlikely to have been done
in actual user code, but would have been trivial to achieve, including simply by
using the `into_iter()` function."*

The 2.5.0 fix introduces `Iterator<'a>` with `PhantomData<&'a ProplistInner>` so
the borrow checker ties the two together, and reworks `into_iter` to transfer
ownership of the C object into the iterator instead of freeing it.

No PulseAudio server is needed: `pa_proplist` is a standalone data structure.

## Why the evidence is valgrind, not AddressSanitizer

`asan.txt` for this row contains a **valgrind memcheck** report. This is deliberate
and is a property of the bug, not a shortcut:

The stale dereference happens inside `libpulse.so` / `libpulsecommon.so` —
distribution shared libraries built without sanitizer instrumentation. ASan only
checks loads and stores it instrumented at compile time. It does poison the region
when `free()` runs, but it never sees `pa_proplist_iterate()` read it, so the
ASan-instrumented build **exits 0**. Valgrind instruments at the machine level and
catches the access wherever it originates.

`run.sh` runs both, and shows the clean ASan pass explicitly so it is not mistaken
for "the bug does not reproduce". Rebuilding all of PulseAudio with ASan would let
ASan see it too, but that is far outside the stock-toolchain bar the task sets and
would not change what the defect is.

This is worth noting for the benchmark generally: for Rust→C rows where the stale
dereference lands in a prebuilt C library, ASan is structurally blind and a
valgrind-class tool is required.

> **Supersedes an earlier SKIPPED filing that targeted the wrong crate.** The
> previous artifact depended on `hlua = "0.1.0"` — a Lua-in-Rust binding — and its
> rationale was that "compiling older `hlua` crates under a modern stable Rust
> toolchain is completely unsupported and unbuildable". But the task spec §6 and the
> benchmark table both define Row 3 as **libpulse-binding** ("Rust→C UAF: iterator
> outlives the C object it reads"), a different crate and a different bug class.
> `hlua` was never the target. The old `build.sh` also omitted `cd "$DIR"`, so
> `cargo build` ran in the parent directory and failed with "could not find
> Cargo.toml" — masked by a trailing `|| true`. Nothing was ever exercised.
