# ldbus #20 — iterator userdata ⟷ freed `DBusMessage` use-after-free

**One line.** A `DBusMessageIter` userdata references a C `DBusMessage` (a reply)
without holding a refcount; when the reply's own Lua wrapper is GC'd
(`dbus_message_unref`), the iterator dangles and reads freed message memory.

## Identity

| | |
|---|---|
| Library | [`ldbus`](https://github.com/daurnimator/ldbus) (daurnimator) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4) |
| Upstream | https://github.com/daurnimator/ldbus/issues/20 (fix PR #21) |
| Vulnerable commit | **`2571a9b`** (parent of the fix) |
| Fix commit | **`5cc933b`** — "Maintain reference to underlying DBusMessage" |
| Native dep | libdbus-1 (verified 1.16.2) + a session bus |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the iterator userdata (wraps a `DBusMessageIter`).
2. **Separate native resource:** the C `DBusMessage` reply (its own Lua wrapper,
   collected because it is unstored in the chain).

The C `DBusMessageIter` points into the message but takes **no refcount** (per the
fix commit), so the message is freed while the iterator lives.

**Direction:** GC-frees. Collecting the reply wrapper `dbus_message_unref`s the C
message; the iterator then derefs it.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), libdbus 1.16.2, private `dbus-run-session`
  bus, gcc 15 ASan. Method substituted to `org.freedesktop.DBus.ListNames`
  (universally available; same reply-lifetime bug as the issue's `Debug.Stats`).
- **Detection note:** libdbus *pools* freed message headers, so a stale read does
  not trap under ASan/valgrind — it stays readable. We force the pooled slot to be
  reused (200 fresh messages), then read the iterator.
- Vuln `2571a9b`: `iter:get_arg_type()` → **nil** (reads the reused empty message).
- Control, fixed `5cc933b`: → **a** (array — the iter keeps the reply alive).
- `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln prints `arg_type=nil`; fixed prints `arg_type=a`. Both required.
