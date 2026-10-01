# memcached safety on QEMU: three heap arms, ten fixtures (2026-10-01)

**Question.** The memcached domain serves its protocol correctly under capability enforcement: M5
gave transcripts identical to native on all three heap arms. What does it do with a heap overflow, a
use after free, a stale free, or the same on one of memcached's own slab items? This is the port
plan's Safety milestone.

**Pre-registration.** The fixtures (`src/mcapp-safety.c`), the hook (patch 0005), the runner
(`host/run-safety.py`) and the predictions (`host/safety-expect.txt`) were pushed to lane branch
`memcached-app` in 92a837b435f4 at 16:51:17. The first fixture boot took the VM lock at 16:52:06.
Nothing in the predictions file has changed since.

## Verdict

**Every run is as pre-registered: 90 of 90.** That is 10 fixtures × 3 arms × 3 repeats; each repeat
is its own boot of the same image (`SHA256SUMS`).

| fixture | level0 | shrink | sublet |
|---|---|---|---|
| 1 malloc(64): the pointer's length | returns; the pointer reaches the arena's end (65,368,224 bytes from its cursor) | returns; **exactly 64** | returns; **exactly 64** |
| 2 write into the neighbouring object | returns `0xee`: the neighbour was overwritten | **bounds fault** at the neighbour's first byte | **bounds fault** |
| 3 read one byte past the end | returns `0x40`, a byte of the next block's header | **bounds fault** at `p + 64` | **bounds fault** |
| 4 read after free | returns the freed object's own byte | returns the same: no temporal safety | **temporal fault** at the freed address |
| 5 read after free and reuse | returns the new occupant's byte, same address | returns the same | **temporal fault** |
| 6 stale free, then allocate | the stale free released the live block; the next malloc aliases it and overwrites it | the same | **temporal fault inside `free`**, on its one-byte probe of the pointer |
| 7 one past a global (control) | bounds fault | bounds fault | bounds fault |
| 8 one past a stack array (control) | bounds fault | bounds fault | bounds fault |
| 9 slab item: write into the neighbouring item | returns `0xee`: the neighbour item overwritten | returns `0xee` | returns `0xee` |
| 10 slab item: read after removal and reuse | returns the new item's byte, same address | the same | the same |

**What is new here, and what is carried over.**
- Fixtures 1–8 and their 24 predictions are the tshark port's (`ports/wireshark/app/host/safety-expect.txt`),
  carried over unchanged; this run confirms them inside memcached, with the fixture running on a
  worker.
- Fixtures 9 and 10 are new: memcached's own allocator, the slabs every stored value lives in. They
  return on every arm, including sublet. A slab page is one `malloc` (`slabs.c:613-618`, no `-L`):
  - shrink and sublet bound each item to its 1 MiB page, not to the item
    (`bounds=[c079f600,c089f600)` and `[c8800000,c8900000)`);
  - level0 bounds it to the whole arena;
  - the slab never calls `free`, so Sublet never revokes.

  That is the gap an in-app Sublet port of the slabs (the plan's stretch S1) would close.

**One premise in the predictions file is wrong, and its predictions held anyway.** It says every
slab item "carries the PAGE's capability, on every arm". On level0 an item carries the whole arena
(`a cursor=c089f4dc bounds=[c02e0530,c42e0530)`), because level0 narrows nothing. The predicted
RETURNs held on level0 because of the arena bounds, and on shrink and sublet because of the page
bounds the premise describes. The file keeps what was predicted.

## How the fixtures ran

- Each fixture is its own server process. `mc-harness --fixture N` starts the arm's safety image
  under `capstone-job` as `nobody`, connects, and sends the hidden `mc_capstone_fixture N`.
- The command is handled where every command is: in `process_command_ascii`, on the worker whose
  event loop owns the connection. Main only accepts and calls `dispatch_conn_new`
  (`memcached.c` `conn_listening`; `thread.c` `thread_libevent_process` → `conn_new`). That is by
  construction.
- **Runtime corroboration is partial.** On level0 and shrink, fixture 8's stack array lies in arena
  memory, not in the monitor-built main stack (`docs/plans/delegation-threads.md`). On sublet it was
  not tied to a region.
- A fixture that returns prints its mark and exits with it. A fault is a domain SIGSEGV with the
  launcher's fault record.

**Judging.** `host/run-safety.py` uses `ports/common/application/check-safety.py`'s own `classify`
and `matches`, imported, on the fixture's stdout and the slice of `qemu.log` written during it.
Attribution rests on three things:
- the faulting address, or the untagged value, equals the target the fixture printed;
- the fixture printed no `returned` line;
- the fault record's pc matches QEMU's diagnostic.

The classifier's "after the touch line" check is structural only: the diagnostics are appended after
the fixture's output. An independent audit resolved every fault pc:
- `mcapp_fix_touch` or `mcapp_fix_poke` for every fixture;
- for sublet fixture 6, `sh_free+0x40`, the probe read.

The audit also doctored inputs to show each of the classifier's gates fires.

## Instrument notes

- The guest harness was rebuilt per run from `host/mc-harness/mc-harness.c`.
  - Level0 and shrink repeat 1 ran c63e9def….
  - Every other boot ran 5457112b…. That build adds a per-connection `dead` flag, which is never
    read on the `--fixture` path.
  - The sublet repeat-1 harness was compiled from that edit 38 s before it was committed
    (8577ff0c0041).
- An earlier build of the safety images (16:47) has different hashes. It predates an edit to
  `mcapp-safety.c`: prototypes for the two access functions, and the slab fixtures' item line split
  from their bounds line. No fixture ran on those images.
- Guest tools in every boot: `capstone-exec` e7e27f49… (the launcher dev pins; the fixtures use one
  connection, so the msghdr race on lane `launcher-msg-call-reentrant` cannot arise) and
  `capstone-job` 5a160efa….

Files: `result-lines.txt` (every run's lines), `SHA256SUMS` (the three images, verifiable from
`$MC_WORK`).
