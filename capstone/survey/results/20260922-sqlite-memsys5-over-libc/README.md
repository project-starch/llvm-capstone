# memsys5's own counters, and the one share this instrument cannot give

`sqlite-memsys5` on SQLite 3.53.3, the revision the port pins. The level's
own traffic is measured. Its reuse share against the system allocator is
**not measurable with this instrument**, and the reason is specific rather
than a shrug.

## Measured

| quantity | value |
|---|---|
| memsys5 allocations | 1,693,135 |
| releases | 1,693,134 |
| reallocs | 24,944 |
| distinct addresses | 389,532 |
| most objects at one address | 198,299 |

Three repetitions, identical.

## Why the share is not measurable here

`memhook.c` watches one boundary, SQLite's `sqlite3_mem_methods`, and the
memsys5 heap never crosses it:

* `speedtest1.c:3218` — `pHeap = malloc(nHeap)`, a plain libc malloc before
  any work.
* `speedtest1.c:3220` — `sqlite3_config(SQLITE_CONFIG_HEAP, pHeap, ...)`
  installs memsys5 over that buffer, replacing the mem methods.
* `speedtest1.c:3475` — `free(pHeap)`, after all of it.

So in this arm level 0 records nothing at all, and its report carries no
`libc` level. That is not an instrument fault: the `shipped` arm of the same
pass records `hook_alloc[libc]` = 1,801,880, so the level is recorded
whenever the mem methods are the default ones. It is the configuration that
moves the boundary out of view.

**What the source shows, which is not the same as a measurement.** One
allocation before the work and one release after all of it, so every memsys5
reuse necessarily precedes the release. That is read off the workload's
source. It is almost certainly the same answer a measurement would give, and
it is not one, and the bundle says which it is.

**What would measure it.** Interposing the malloc family, which the APR and
memcached wirings do and `memhook.c` does not. That is a change to an
instrument three passes share, x86, CheriBSD and the Capstone domain, so it
is worth doing deliberately rather than for a row whose answer is already
visible in eight lines of the workload.

## Relation to the level above it

`sqlite-lookaside` sits on memsys5 and its bundle beside this one answers
the nested question properly, with a stamp clock and a diagnostic that
separates "no stamp" from "a stamp too late". That level's answer is
100 per cent and it is measured. This level's answer is the same number
and it is inferred, and the two should not be quoted as though they were
the same kind of thing.

## Provenance

| | |
|---|---|
| SQLite | 3.53.3, pinned in `experiments/a1/sources.sha256` |
| workload | `--heap 268435456 64 --memdb --size 100 --verify --stats` |
| companion pass | `experiments/a1/results/x86/20260921T213027Z` |
| raw | `raw/`, hashed in `raw/SHA256SUMS`, with the `shipped` arm kept as the control that shows level 0 is recorded elsewhere |
