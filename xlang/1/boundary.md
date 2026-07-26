# Language Boundary Violation — rlua #19 (Row 1 under Tier 3)

### Boundary Pointer
The Rust struct object pointer (wrapping the heap-allocated userdata `Userdata`) crosses the language boundary. It is allocated and dropped in Rust space but is wrapped and passed as GC-managed userdata into Lua space.

### Lifetime Violation Details
In the vulnerable version of `rlua`, the userdata destructor was implemented using `mem::replace(obj, mem::uninitialized())`. 

When Lua performs garbage collection, it runs the `__gc` finalizer of the userdata object, crossing the language boundary back to Rust space, which drops/frees the Rust object. However, inside the Lua `__gc` finalizer, the Lua script resurrected the userdata reference by assigning it to a global variable `hatch`.

Because `rlua` stored `T` directly and replaced it with `mem::uninitialized()` upon drop, the Lua-held resurrected reference `hatch` remains active but carries a dangling pointer to deallocated/uninitialized memory. Subsequent method calls from Lua (such as `hatch:access()`) cross the language boundary back to C/Rust and read from this uninitialized memory (retrieving garbage data) or trigger a double-free on the next GC cycle.
