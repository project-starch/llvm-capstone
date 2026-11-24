# pymalloc component on the virtual profile, 2026-11-11

The port's `replay` and `allocator-example` as Capstone processes, `PYMALLOC_SUBLET` on and off.
On is patches 0001 to 0004 (`prepare-source.py --variant protected-virtual`): every block a child
lifetime of its arena, ended with the one-operand `CREVOKE`.

The replay ran the trace of the native suite (`tests/native/test-replay.py`, 25,409 events: every
size class, pool reassignment, raw fallback, arena release) and was judged by that test's
assertions on the report file, with the native run of the same date as the reference for the
counts and the event checksum.

| program | `PYMALLOC_SUBLET=ON` | `PYMALLOC_SUBLET=OFF` |
|---|---|---|
| `replay` | exit 0; completed 25,409, alloc 12,176, free 12,176, realloc 1,056; report as native | the same |
| `allocator-example` | exit 0, `ALLOCATOR_EXAMPLE cpython PASS pointer_bytes=16` | the same |

Both replays use 7 arenas and release 6. The allocation-decision hash differs between the two
builds and from the native one; the gate does not compare it.

The native suite of the same tree: `allocator-replay` and `source-preparation` pass.
`port-support` runs 54 tests and one errors, `test_a_rebuilt_sdk_is_invisible_at_head`, which asks
`git log` for a file's revision and ran in a copy of the tree without history.

## Platform

The tree's virtual kit on the processor model with the one-operand `CREVOKE` and `CRENEW`;
`result.json` carries the file hashes, the image hashes and the result lines. CPU properties:
`x-capstone-test-csrs=true,x-capstone-exact-bounds=true` on top of the staged runner's defaults.
The module was loaded without parameters (`unpinned=0`, `supervisor_slots=1`). One boot ran the
gates of the CPython application, the pymalloc component and the wmem component, ten processes
under `capstone-vexec`. After the last one `capstone-vexec --stats` reported 0 live domains, 0 live
regions, 0 live bytes and 0 tag pages, and the console holds no domain fault.
