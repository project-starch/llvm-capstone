/* lua-SDL2 #75 — Lua userdata ⟷ C SDL_Window use-after-free.
 * Source: ../../lua-sdl2-75/boundary.md. valgrind: Invalid read of size 8, 0
 * bytes inside a 240-byte SDL_Window freed by SDL_DestroyWindow.
 *
 * Two allocations: two CommonUserdata window handles (A, B) that alias ONE
 * native SDL_Window (the bug duplicates the handle, both mustdelete=1).
 *   Free-site (window.c:1002): collectgarbage() finalizes duplicate B ->
 *     l_window_gc -> SDL_DestroyWindow(udata->data) frees the 240-byte window.
 *   Stale-use (window.c:267): handle A still live -> win:getID ->
 *     l_window_getID -> SDL_GetWindowID reads window->magic on the freed block.
 * READ size 8 at OFFSET 0 -> a plain load through the revoked capability
 * (clean cause-25 route). Control: the read returns and the row reports MISS.
 */
#include "luac_shim.h"
#include <stdint.h>

#define SDL_WINDOW_BYTES 240

static volatile uint64_t sink;

int main(void) {
  void *window = malloc(SDL_WINDOW_BYTES); /* SDL_CreateWindow */
  if (!window)
    abort();
  memset(window, 0, SDL_WINDOW_BYTES);

  void *handle_A = window; /* the still-live script/event-loop handle */
  void *handle_B = window; /* the duplicate the GC finalizes            */

  free(handle_B);          /* l_window_gc -> SDL_DestroyWindow -> REVOKE */

  /* win:getID on A -> SDL_GetWindowID reads window->magic at offset 0. */
  sink = *(volatile uint64_t *)handle_A; /* window.c:267 Invalid read size 8 */

  mock_report("luac_sdl_window_uaf", "use-after-free-survived");
  return 0;
}
