# cffi-lua #57 — Lua cdata ⟷ libffi `closure_data` use-after-free

**One line.** A `cffi.cast` callback cdata wraps a separate heap `closure_data`
(libffi closure); `callback:free()` frees the C struct without nulling the
cdata's pointer, so `callback:set()` reads the freed block (and a second
`:free()` double-frees it).

## Identity

| | |
|---|---|
| Library | [`cffi-lua`](https://github.com/q66/cffi-lua) (q66) |
| Language pair | **C/C++ ⟷ Lua** (reference Lua 5.1–5.4). *Weakest cross-language case of the set — the coupled object is FFI bridge plumbing, not a third-party library resource; reproduces in pure Lua.* |
| Upstream | https://github.com/q66/cffi-lua/issues/57 |
| Vulnerable commit | **`d295f029a72fa2544eefe0bdbb40d2af8f4f18dc`** (parent of the fix) |
| Fix commit | **`ced2cba79`** — "prevent :set() on callbacks that had :free() called on them" |
| Native dep | libffi (verified 3.5.2) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the `callback` cdata from `cffi.cast("void (*)()", fn)`.
2. **Separate native resource:** a 56-byte `closure_data` C++ struct (`new[]` in
   `make_cdata_func`, `ffi.cc:271`) holding the libffi `ffi_closure`.

Two distinct allocations. **Not** the string-borrow shape.

**Direction:** native-frees. `:free()` frees the `closure_data` (crossing 1);
`:set()` derefs the stale wrapped pointer (crossing 2).

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: PUC Lua 5.4.7 (built shared from source), libffi 3.5.2, gcc 15 ASan.
- Vulnerable `d295f029`: ASan **heap-use-after-free**, `READ` in
  `cdata_meta::cb_set` (`ffilib.cc:281`), freed by `ffi::destroy_closure`
  (`ffi.cc:127`) via `cb_free` (`ffilib.cc:268`), block alloc'd in
  `make_cdata_func` (`ffi.cc:271`).
- Control, fixed `ced2cba79`: **no ASan report** — a clean `bad callback` Lua
  error instead.
- Full trace + control in `evidence.txt`.

## PASS signature

`run.sh` passes iff, on the **vulnerable** commit, ASan reports
`heap-use-after-free` with `cdata_meta::cb_set` (`ffilib.cc:281`) as the use
site AND `ffi::destroy_closure`/`cb_free` as the free site — AND the **control**
run on the fixed commit produces no ASan report. Either half missing = FAIL.
