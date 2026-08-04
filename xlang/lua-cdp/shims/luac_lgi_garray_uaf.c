/* lgi #65 — Lua guard userdata ⟷ C GArray use-after-free.
 * Source: ../../lgi-65/boundary.md. valgrind: invalid READ size 8, 0 bytes
 * inside a 16-byte GArray freed by g_array_unref.
 *
 * Two allocations: the lgi guard userdata (lgi_guard_create) and the GArray
 * (g_array_sized_new) whose data segment is installed into the C struct field
 * DBusInterfaceInfo.methods.
 *   Free-site (core.c:256): collectgarbage() collects the unanchored guard ->
 *     guard_gc -> g_array_unref frees the 16-byte GArray + its data.
 *   Stale-use (marshal.c:562): print(iface.methods) -> lgi_marshal_field ->
 *     marshal_2lua_array walks the freed GArray, reading ->len first.
 * READ size 8 at OFFSET 0 (the ->len word) -> plain load through the revoked
 * capability (clean cause-25). Control: the walk returns and the row reports.
 */
#include "luac_shim.h"
#include <stdint.h>

#define GARRAY_BYTES 16 /* the GArray header valgrind names */

static volatile uint64_t sink;

int main(void) {
  void *garray = malloc(GARRAY_BYTES); /* g_array_sized_new */
  if (!garray)
    abort();
  memset(garray, 0, GARRAY_BYTES);

  /* iface.methods borrows the GArray's memory; the guard owns its free. */
  void *iface_methods = garray;

  free(garray); /* guard_gc -> g_array_unref -> REVOKE */

  /* marshal_2lua_array reads ->len at offset 0 of the freed GArray. */
  sink = *(volatile uint64_t *)iface_methods; /* marshal.c:562 invalid read */

  mock_report("luac_lgi_garray_uaf", "use-after-free-survived");
  return 0;
}
