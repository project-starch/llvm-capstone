# sol2 #1080 — Lua `sol.Foo*` pointer-userdata ⟷ destroyed native C++ Foo use-after-free

**One line.** Passing an lvalue C++ object to Lua makes sol store a **`sol.Foo*`
pointer-userdata** (a reference, not a copy); the native `Foo` is then destroyed
while Lua keeps that userdata alive (global `test`), and a later `test:Print()`
re-derefs the freed object.

## Identity

| | |
|---|---|
| Library | [`sol2`](https://github.com/ThePhD/sol2) (ThePhD) |
| Language pair | **C++ ⟷ Lua** |
| Upstream | https://github.com/ThePhD/sol2/issues/1080 (owner-confirmed by ThePhD) · repro godbolt aG3qax (buggy) / Yffz4o (value-copy) |
| Version | header-only; **v3.5.0** (the reference-vs-value push semantics are version-stable; issue filed against 3.2.1) |
| Native dep | none beyond a C++17 compiler |

## The two coupled objects (why unambiguous CDP)

1. **Lua-side handle:** the `sol.Foo*` **pointer-userdata** created when the lvalue
   `Foo` is pushed to Lua; stored in the Lua global `test`, it outlives the C++
   object. Confirmed a pointer wrapper by `debug.getmetatable(foo).__name` ==
   `sol.Foo*` (the value-copy control reports `sol.Foo`).
2. **Native C++ object:** the heap `Foo` (`new Foo()` … `delete m`). The pointer-
   userdata stores its address; sol does **not** copy it.

Two distinct allocations. **Not** a string-borrow — the freed memory is the native
`Foo` storage, and the Lua userdata (which survives) is a separate allocation.

**Direction:** NATIVE-frees. `delete m` frees the native `Foo`; the Lua userdata
lives on, and re-dereferencing it (`test:Print()`) reads the freed block.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), sol2 v3.5.0, gcc 15 ASan.
- Vuln: ASan **heap-use-after-free**, `READ of size 4` in `Foo::Print()`
  (`harness.cpp:39`) reading `this->val` of the freed native `Foo`, reached
  through the full sol member-dispatch chain
  (`member_function_wrapper` → `lua_cfunction_trampoline` → `luaD_pretailcall`).
  Freed by `delete m` (`:67`); allocated by `new Foo()` (`:62`).
- Control (rvalue push `storeFoo(Foo(*m))` → Lua-owned `sol.Foo` value copy):
  clean, `DONE`, no ASan — the copy survives the identical `delete m`.
- Trace + control in `evidence.txt`; `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln run: ASan `heap-use-after-free` whose faulting frame is `Foo::Print` (the
Lua→C++ method dispatch on the dangling `sol.Foo*` userdata). Control run: `DONE`,
no `AddressSanitizer` report. Both required (run.sh asserts both, exits nonzero
otherwise).
