# Evidence behind RESULTS.md that is not a result line

## Chronology: the run host's clock

The `started_utc`/`finished_utc` fields in `runs/*/run.json` are the run host's clock, and that clock
is slow: measured on 2026-10-24 at 11:38 UTC, three samples of `date +%s.%N` on both machines bracketed
by the SSH round trip (0.18 s) put the run host 1597.5 s behind the host that made the commits, with
NTP inactive on the run host (`timedatectl`: "System clock synchronized: no"). Assuming that offset held
through the day:

| event | recorded | on the commit host's clock (UTC) |
|---|---|---|
| `PREDICTIONS.md` committed, 3cd43d008369 | 16:54:07 +08:00 | 08:54:07 |
| `published-la` / `published-nola` started | 08:27:47 / 08:27:48 | 08:54:25 |
| the same finished | 08:32:50 / 08:32:51 | 08:59:28 |
| `PREDICTIONS-REVIEW.md` committed, 378c8ea8f8bf | 16:59:40 +08:00 | 08:59:40 |
| `published-*-rep2` started | 08:33:41 / 08:33:42 | 09:00:19 |

So the predictions were committed about 17 s before the first runs started, and the review about 12 s
after the first pair finished and before the repetitions started. That the review was written before any result
was read is its author's statement; the timestamps allow it and cannot show it.

## Heap arms: only the heap size differs

`runs/heap8-la` and `runs/heap32-la` were built by `corpus/build-cases.py --heap-bytes`. Against the
`published-la` build, on the run host: `src/sqlite3.c` is byte-identical (`cmp`); the engine compile
commands (`engine-core.log`, `engine-fts3.log`, `engine-rtree.log`, first line) are identical once the
build directory's own name is removed; the case compile commands for 07, 21 and 23 differ by exactly one
argument, `-DSQLITE_HEAP_SIZE=8388608` or `-DSQLITE_HEAP_SIZE=33554432`.

## Case 15: where their QEMU reads the poison marker

Their QEMU source is the pinned base (CTSRD-CHERI/qemu 5571e801bf5b) with the artifact's files laid over
it by `ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/prepare.py`. The two files below, as built, are
byte-identical to the artifact's copies (`Ajs38RO4Vz-p-731c572/qemu/target/cheri-common/`):

- `cheri-helper-utils.h:387`, in `cap_check_common_reg`, and `op_helper_cheri_common.c:1841`, in
  `cgetpoison`:
  `unsigned int mask = ~((1 << 0) | (1 << 2) | (1 << 3)| (1 << 4));`
- used at `cheri-helper-utils.h:393` and `op_helper_cheri_common.c:1844` as
  `probe_read(env, addr & mask, CHERI_CAP_SIZE, ...)`, the read of the capability-sized word that holds
  the marker.

The mask clears bits 0, 2, 3 and 4 and keeps bit 1 (and, being a 32-bit `unsigned int`, clears bits 32-63
of the address, which does not matter for the low heap addresses here).

The mask is already in the pinned base: it came in with CTSRD-CHERI/qemu d4441202f4fc (2025-09-11, "store
a dummy value on poison, and only throw trap if the memory contains it"), whose patch adds it to
`cheri-helper-utils.h` with `probe_read(env, addr&mask, ...)`. The repository's `poison-cap-smode` branch
(head 5b90407885fb, 2025-09-17) reads that word with `probe_read(env, QEMU_ALIGN_DOWN(addr,
CHERI_CAP_SIZE), ...)` (`cheri-helper-utils.h:387` on that branch). Their Toooba RTL, from the artifact
(`Toooba/src_Core/RISCY_OOO/coherence/src/`), selects the word with `getLineMemDataOffset(req.addr)`
(`L1Bank.bsv:615`), defined as `truncate(a >> valueOf(LgMemDataSzBytes))` (`CCTypes.bsv:127-128`), a
shift with no mask on bit 1. Neither the branch nor the RTL was built or run here.

The measurement this explains is `runs/ev3-la`, control `uaf_offsets`
(`ports/sqlite/poisoncap/corpus/controls/uaf_offsets.c`): for each offset, a fresh child process allocates a
128-byte block (a 100-byte request), frees it, and reads one byte at that offset through a pointer
derived before the free. Offsets 0-31, 55
and 127: SIGPROT (code 3) at exactly the 16 offsets with bit 1 clear, a normal exit at the 18 with bit 1
set, 34 of 34. Case 15's stale read is at 0x236c37, which has bit 1 set (`runs/diag3-la`). The same run
read offset 0x37 of a freed block twice more, after a read through the same pointer before the free
(`uaf_read_before_free`) and without one (`uaf_interior`): both returned, so the earlier read is not what
lets case 15's access through; offset 0x37 has bit 1 set either way.
