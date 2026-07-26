# rlua #19 (Row 1 under Tier 3) — Use-After-Free / Double-Free in rlua

This is a minimal, deterministic reproduction of `rlua #19` (Row 1 under Tier 3 in `xlang-repro-task.md`), a heap Use-After-Free/Double-Free of userdata in `rlua` via garbage collection resurrection.

## Vulnerability Overview
In `rlua`'s handling of Lua's `__gc` metamethod, a userdata object can be "resurrected" during Lua GC (by assigning a reference of it to a global variable within the `__gc` function). Because the vulnerable `rlua` destructor forcefully drops the underlying Rust object using `mem::replace(obj, mem::uninitialized())` upon the first `__gc` call, the resurrected Lua reference `hatch` is left pointing to uninitialized/freed memory. Any subsequent access from Lua reads garbage values (Use-After-Free) and a second GC cycle on `hatch` results in a double-free of uninitialized memory.

## Contents
* `target.md` - Pinned versions and commit metadata
* `Cargo.toml` - Rust project configuration with local `rlua` dependency
* `src/main.rs` - Rust entrypoint demonstrating the resurrection and UAF access
* `build.sh` - Automated build script to clone and build rlua at the vulnerable commit
* `run.sh` - Runs the trigger natively

## How to Build and Run
To build the vulnerable rlua and compile targets natively:
```bash
chmod +x build.sh run.sh
./build.sh
```

To run the reproduction and capture output:
```bash
./run.sh
```

## Expected Outcome
The execution of the trigger script will output garbage data instead of the original value `123` on access (e.g. printing `accessing userdata 32`), confirming the use-after-free.
