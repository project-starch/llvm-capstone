# Boundary annotation — sol2 #1080

### The object that crosses the boundary

A `sol.Foo*` **pointer-userdata** created when a C++ **lvalue** `Foo` is pushed to
Lua. It stores the raw address of the native `Foo`; sol makes no copy. It crosses
from C++ into Lua and is parked in the Lua global `test`, so it outlives the native
object's scope.

### Owner vs. borrower

- **C++ (native) owns the storage.** The `Foo` is a native heap allocation
  (`new Foo()` / `delete m`); its lifetime is the C++ code's, not Lua's.
- **Lua borrows a raw pointer** to it, via the `sol.Foo*` userdata. There is no
  lifetime tie back to the native object — Lua holds an address, not a copy.
- sol pushes a **registered-usertype lvalue by reference (pointer), not by copy**
  (owner-confirmed default). An rvalue (`Foo{}` / `Foo(*m)`) is instead copied into
  a Lua-owned `sol.Foo` value userdata — that is the control, and the fix.

### Free site

`delete m` (harness.cpp:67) frees the native heap `Foo`. (Upstream: end of the
`{ Foo m; … }` block runs `~Foo` — a stack unwind; we heap-allocate so ASan sees a
heap free, but the lifetime relationship is identical.)

### Stale-use site (one crossing later)

`test:Print()` (harness.cpp:70) → sol reads the still-alive `sol.Foo*` userdata,
recovers the now-dangling `Foo*`, and dispatches `Foo::Print` on it
(`member_function_wrapper::call` → trampoline → `luaD_pretailcall`). `Foo::Print`
reads `this->val` → heap-use-after-free READ of the freed native `Foo`
(harness.cpp:39).

### The lifetime rule that is violated

A Lua handle that stores a raw pointer to a native object must not outlive that
object; either the native side must pin the object for the userdata's lifetime, or
the value must be copied across the boundary. sol's default lvalue push does
neither — "the C++-isms leak through" — so the `sol.Foo*` userdata outlives the
`Foo` it points at.

### Capability note (revoke-on-free)

Revoke-on-free revokes the capability to the native `Foo` block at `delete m`. The
address held inside the `sol.Foo*` userdata is then a revoked capability, so the
`test:Print()` re-deref faults at the contract point (the method dispatch) instead
of reading freed native bytes — turning a silent UAF into a deterministic trap
exactly where the boundary is crossed.
