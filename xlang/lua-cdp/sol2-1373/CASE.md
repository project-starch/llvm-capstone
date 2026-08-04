# sol2 #1373 — Lua interior-reference ⟷ GC'd C++ struct use-after-free

**One line.** `bb = csc.b` yields a Lua wrapper holding an **interior pointer**
into a `ComplexStructC` owned by a Lua-GC userdata; collecting `csc` frees the
storage, and reading `bb.a.a` dereferences the freed interior.

## Identity

| | |
|---|---|
| Library | [`sol2`](https://github.com/ThePhD/sol2) (ThePhD) |
| Language pair | **C++ ⟷ Lua** |
| Upstream | https://github.com/ThePhD/sol2/issues/1373 (owner-confirmed as-designed C++-ism) · repro godbolt z/Eb6zGaKjr |
| Version | header-only; **v3.5.0** (v3.3.0 from the issue no longer builds on gcc 15; the behaviour is version-stable) |
| Native dep | none beyond a C++17 compiler |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the userdata for `csc` (a `ComplexStructC` returned by value).
2. **Separate reference:** `bb`, a sol2 interior-reference wrapper pointing into
   `csc`'s storage (proven a reference, not a copy, because `bb.b = 100` is
   visible via `csc.b.b`).

Two distinct allocations. **Not** a string-borrow — the freed memory is the C++
`ComplexStructC` storage.

**Direction:** GC-frees. `csc = nil` + collect frees the parent userdata; reading
`bb.a.a` derefs the freed interior.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), sol2 v3.5.0, gcc 15 ASan.
- Vuln: ASan **heap-use-after-free**, `READ` in
  `sol::stack::unqualified_pusher<int>::push` (`stack_push.hpp:316`), reading the
  interior `ComplexStructA::a` after the parent userdata was collected.
- Control (keep the parent referenced via `keep=csc`): clean, `DONE`, no ASan.
- Trace + control in `evidence.txt`; `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln run: ASan `heap-use-after-free` with a `sol::stack::…push` frame reading the
interior member. Control run: `DONE`, no ASan report. Both required.
