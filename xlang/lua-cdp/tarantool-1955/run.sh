#!/usr/bin/env bash
# tarantool #1955 is BLOCKED in this sandbox: the bug is ASan-only (a read of a
# freed struct error, not a call through a nulled vtable), so it does not fault
# on stock release/Docker images; and the issue ships no minimal reproducer.
# Observing it needs a full from-source ~1.10.2 build with AddressSanitizer
# (bundled LuaJIT + 2018-era submodules) -- impractical here. Upstream-verified
# via the filed ASan trace (see evidence.txt); mechanism source-confirmed in
# boundary.md. LuaJIT cdata -> for a reference-Lua Capstone vehicle, reproduce
# in userdata form.
echo "BLOCKED: tarantool #1955 -- ASan-only UAF, no minimal repro; needs a"
echo "from-source Tarantool 1.10 + LuaJIT ASan build. See CASE.md / evidence.txt."
exit 2
