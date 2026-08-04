# Boundary annotation — lua-SDL2 #75

### The object that crosses the boundary

The raw `SDL_Window*` (a 240-byte native struct owned by libSDL2), stored in the
`data` field of a Lua `CommonUserdata`. The userdata is the Lua-visible Window
handle; the `SDL_Window*` is what crosses into C. The bug is that *two* userdata
end up holding the *same* `SDL_Window*`, both with `mustdelete = 1`.

### Owner vs. borrower

- **libSDL2 owns the native window.** `SDL_CreateWindow` (`l_createWindow`,
  `window.c:76`) allocates it; `SDL_DestroyWindow` frees it.
- **Lua (managed) owns each handle.** A Window userdata's lifetime is the GC's; its
  stored `data` pointer is the coupling. `mustdelete=1` means "this handle owns the
  window and must destroy it on `__gc`."
- The bug: `SDL.createWindow` returns handle **A** (`mustdelete=1`), the legitimate
  owner. On the vulnerable tree the hit-test callback then mints handle **B** over
  the *same* window, *also* `mustdelete=1`. Two owners of one resource. Whichever is
  collected first destroys the window out from under the other (and the event loop).

### Free site

`collectgarbage()` finalizes the unreferenced duplicate **B** →
`l_window_gc` (`src/window.c:1002`) → `SDL_DestroyWindow(udata->data)` → the native
240-byte window is `free`'d, `window->magic` nulled.

### Stale-use site (one crossing later)

Handle **A** is still live (held by the script and used by the event loop). The
next `win:getID()` → `l_window_getID` (`src/window.c:267`) → `SDL_GetWindowID` reads
`window->magic` on the freed block → valgrind **`Invalid read of size 8`**; SDL's
own `CHECK_WINDOW_MAGIC` then reports **`Invalid window`**.

### The lifetime rule that is violated

Exactly one handle may own a native resource. A view/borrow handle created over an
already-owned resource must not carry the delete flag. The fix (`96491c0`) builds
the hit-test's handle with `commonPushUserdata` and sets `cu->mustdelete = 0`, so
collecting it is a no-op and the single owner (handle A) remains valid.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, the `SDL_DestroyWindow` free at `window.c:1002`
**revokes** the capability to the `SDL_Window`. The surviving handle A still names
that capability, so the read at `window.c:267` (`SDL_GetWindowID`) faults at the
contract point — the delivered fault the capability model promises, in place of the
valgrind-detected UAF / SDL's after-the-fact "Invalid window" guard.
