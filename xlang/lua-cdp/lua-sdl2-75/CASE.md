# lua-SDL2 #75 — Lua Window userdata ⟷ native `SDL_Window` use-after-free

**One line.** The window hit-test callback builds a SECOND Window userdata over
the SAME native `SDL_Window` with `mustdelete=1`; when that duplicate is GC'd its
`__gc` calls `SDL_DestroyWindow` on the window the original userdata and the event
loop still use.

## Identity

| | |
|---|---|
| Library | [`lua-SDL2`](https://github.com/Tangent128/luasdl2) (Tangent128) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4). Native third-party resource (an SDL2 window handle), not FFI plumbing — a strong cross-language CDP case. |
| Upstream | https://github.com/Tangent128/luasdl2/issues/75 · fix PR https://github.com/Tangent128/luasdl2/pull/77 (merge `5d3212e`) |
| Vulnerable commit | **`272a748`** (= `96491c0^`, parent of the fix) |
| Fix commit | **`96491c0`** — "fix for hittest bug (#75)": `commonPush(...,"p",...)` → `commonPushUserdata(...)` + `cu->mustdelete = 0` in `src/window.c` |
| Native dep | SDL2 (verified 2.32.10) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** a `CommonUserdata` (the Window object). Its `__gc`
   (`l_window_gc`, `window.c:1002`) does `if (mustdelete) SDL_DestroyWindow(data)`.
2. **Separate native resource:** the `SDL_Window*` from `SDL_CreateWindow`
   (`l_createWindow`, `window.c:76`) — a 240-byte native struct owned by libSDL2.

Two distinct allocations (Lua userdata vs. the native `SDL_Window`). The bug makes
*two* userdata handles alias *one* native window, both flagged `mustdelete=1`.

**Direction:** GC-frees. Collecting the duplicate handle frees the native window
(crossing 1 → native `SDL_DestroyWindow`); the surviving handle then reads the
freed window (crossing 2 → `SDL_GetWindowID`).

## How the duplicate is born

`hitTestCallback` (`window.c:910`) runs on every hit-test. On the vulnerable tree
it does `commonPush(cd->L, "p", WindowName, win)`, and `commonPushUserdata`
defaults `mustdelete = 1`. SDL invokes the hit-test only on a real left
`ButtonPress` inside the window (X11 `ProcessHitTest`), so the trigger drives a
real X button event with `xdotool` under `xvfb`, then `collectgarbage()` finalizes
the now-unreferenced duplicate.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: shared PUC Lua 5.4.7 (built shared), SDL2 2.32.10, gcc 15, headless via
  `xvfb-run` + `xdotool`.
- Vulnerable `272a748`, ASan module: after `collectgarbage()`, `win:getID()`
  returns 0 and `SDL.getError()` == **`Invalid window`** (marker
  `WINDOW-DESTROYED-WHILE-LIVE`) — the live window was destroyed by the duplicate's
  `__gc`.
- Vulnerable `272a748`, valgrind (non-ASan module — libSDL2 is not ASan-built):
  **`Invalid read of size 8`** in `SDL_GetWindowID` ← `l_window_getID`
  (`window.c:267`), on a **240-byte block free'd** by `SDL_DestroyWindow` ←
  `l_window_gc` (`window.c:1002`, from the GC finalizer), **alloc'd** by
  `SDL_CreateWindow` ← `l_createWindow` (`window.c:76`).
- Control, fixed `96491c0`: window survives GC → `WINDOW-ALIVE`, no `Invalid
  window`, valgrind **0 errors**.
- Full traces + control in `evidence.txt`.

## PASS signature

`run.sh` passes iff, on the **vulnerable** build, the trigger fires the hit-test
(`HITCOUNT>=1`) AND reports `WINDOW-DESTROYED-WHILE-LIVE` with SDL error `Invalid
window`; AND the **fixed** build fires the hit-test AND reports `WINDOW-ALIVE`
with no `Invalid window`. When `valgrind` is present it must also show the
`Invalid read` on the 240-byte freed `SDL_Window` on the vulnerable build and `0
errors` on the fixed build. Any half missing = FAIL.
