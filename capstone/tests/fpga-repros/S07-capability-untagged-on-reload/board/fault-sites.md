# The three fault sites, as measured

Bitstream `caplifive_12august.bit`. `mepc` is read from the latched trap state the bitstream exposes
on switches 196-203; `sw=255` carries {seen, mcause[6:0]}. All three read **0x99** = seen, mcause 25.

Image VA is recovered as `(mepc & 0x3FFFFF) + 0x10000`. That is exact here because the domain is
allocated with `__get_free_pages(order)` — a buddy allocation, naturally aligned — and these images
force order 10, i.e. 4 MiB alignment (`modcapstone/module/capstone.c:107-113`); `0x10000` is the
single PT_LOAD's vaddr. **The recovery is size-dependent**: a small domain gets a lower order and
less alignment, and the masking would then be unfounded.

| instance | mepc | VA | site | faulting instruction |
|---|---|---|---|---|
| 1 | `0x83143d70` | `0x153d70` | `memcpy+0x2a8` | `cincoffset a1, a2, a1` after `ldc a2, 0x0(a2)` |
| 2 | `0x835416a8` | `0x1516a8` | `output_text+0xdc` | `cincoffset a1, a1, a2` on the payload capability |
| 3 | `0x83025490` | `0x35490` | `sqlite3DbMallocRawNN+0xd8` | `ldc a0, 0x2a0(a0)` (`db->lookaside.pSmallFree`) |

Instance 1 was reproduced at the identical instruction across two DIFFERENT SQL statements
(`CREATE INDEX`, and a `SELECT count(*)` substituted for it via the `CAPSTONE_EXT_SKIP_INDEX`
control), so it is not statement-specific.

## A caveat on the latch

`mepc` proves where the last capability exception was, not that the core stopped there. Supporting
the identification: the latch keeps the latest trap with `cause ∉ {0,2}` and is cleared only by
reset; Linux runs between tests and every syscall or page fault would overwrite it with a *virtual*
pc — so a surviving cause-25 at a bare-physical `0x83xxxxxx` postdates all inter-test kernel
activity. Twice, on longer-running arms, the latch **was** overwritten (mcause 9 at
`0xffffffff800072cc`, a kernel VA) and carried no information. A kernel VA in that field means NO
DATA, not a finding about the kernel.
