# Record-access traces and exact instruction zones (nested lane, 2026-11-20)

Two instruments for measuring what lifetime protection does, both outside the programs they measure.

## The record-access trace (QEMU twin)

`capstone-qemu-a0501ebc39-record-trace.patch` applies to capstone-qemu `a0501ebc39` (the ids55 native twin, P1's
table ABI) with `git apply`. With `CAPSTONE_RECTRACE=<prefix>` in QEMU's environment, every access the model makes to a
record of a guest lifetime table is appended to `<prefix>.<pid>`, in execution order; without it QEMU behaves as
before (the inline capability-access path is used only when the trace is off). The format is in
`target/riscv/cap_rev_trace.h` and in the docstring of `rectrace-read.py`:

- one little-endian u64 per run of identical events: code (kind << 2 | op), count, slot;
- kinds: check of a data capability in protected code (QEMU's own liveness caches are bypassed, so every dereference
  is an event), check outside it, PCC check, allocation, release, the release's read-only first pass, other;
- markers: table root changes, instruction counts every 1,024 runs (all, and protected code only), context edges of
  protected code (trap out, SRET or MRET back in);
- completed runs are written every 2 s of wall time, so a QEMU killed by its runner loses at most that interval;
- `<prefix>.<pid>.stats` holds the totals per event code.

`rectrace-read.py TRACE` prints the totals and exits 2 on an empty or malformed file; `--text --from N --lines M`
prints runs as lines.

What the model writes that hardware need not: an allocation writes the new record twice (the pop, then the record
with its links), and a release writes each released record twice (valid cleared, then the push). A replay that prices
writes should drop one of each.

## Exact instruction zones (TCG plugin)

`zones.c` builds against `include/qemu/qemu-plugin.h` of the QEMU it is loaded into
(`gcc -O2 -fPIC -shared -I<qemu>/include/qemu $(pkg-config --cflags glib-2.0) -o zones.so zones.c`). It counts every
executed instruction by the address of its code: a native image's window (`img_lo=`, `img_hi=`, hex), other user
code, firmware, kernel; and writes the counts to `out=` when QEMU exits. QEMU 8.1's plugin API has no privilege
query, which is why the zones are by address. On the native task route a process's image is mapped below
0x3ff8000000, so the image zone is that program's own user-mode instructions, exactly, and one program per boot keeps
the zone to one job. The plugin changes no guest-visible state.

Results measured with both are in `capstone/docs/ref/nested-allocator-workloads.md`.
