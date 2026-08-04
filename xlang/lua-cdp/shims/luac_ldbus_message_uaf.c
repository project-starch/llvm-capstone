/* ldbus #20 — Lua iterator userdata ⟷ C DBusMessage use-after-free.
 * Source: ../../ldbus-20/boundary.md. On the real library this does NOT trap
 * under ASan/valgrind: libdbus POOLS freed message headers, so the stale read
 * lands in reused pool memory and is visible only as a value differential
 * (arg_type=nil vs 'a', after forcing pool reuse with 200 fresh messages).
 *
 * Two allocations: the iterator userdata wrapping a DBusMessageIter and the C
 * DBusMessage reply.
 *   Free-site: the unstored reply wrapper becomes unreachable; collectgarbage()
 *     runs its __gc -> dbus_message_unref frees the DBusMessage.
 *   Stale-use (message_iter.c): iter:get_arg_type() ->
 *     ldbus_message_iter_get_arg_type -> dbus_message_iter_get_arg_type reads
 *     the freed message's type tag.
 * READ at OFFSET 0 -> plain load through the revoked capability (clean cause-25
 * route). Control: the read returns; row reports MISS.
 *
 * NOTE: Capstone's revoke-on-free is STRICTER than ASan here — it revokes the
 * block regardless of libdbus's pooling, so the stale read faults where a
 * pooling allocator hid it. Size not named in any trace; a 16-multiple stand-in.
 */
#include "luac_shim.h"
#include <stdint.h>

#define DBUS_MESSAGE_BYTES 64 /* size not named; libdbus pools these */

static volatile uint64_t sink;

int main(void) {
  void *msg = malloc(DBUS_MESSAGE_BYTES); /* the DBusMessage reply */
  if (!msg)
    abort();
  memset(msg, 0, DBUS_MESSAGE_BYTES);

  void *iter_msg = msg; /* the iterator userdata caches the message pointer */

  free(msg); /* __gc -> dbus_message_unref -> REVOKE */

  /* iter:get_arg_type -> dbus_message_iter_get_arg_type reads the type tag. */
  sink = *(volatile uint64_t *)iter_msg; /* message_iter.c */

  mock_report("luac_ldbus_message_uaf", "use-after-free-survived");
  return 0;
}
