# wmem component on the virtual profile, 2026-11-11

The port's `replay` and `allocator-example` as Capstone processes, `WM_SUBLET` on and off, after
patch 0001 changed to the one-operand `CREVOKE` and to `CRENEW` at a reset.

The replay ran the directed trace of the native suite (`tests/native/test-replay.py`, 1,661 events
across all four allocators: allocation, resize, individual free, reset, collection and
destruction, jumbo objects of 9 and 3 MiB resized to 10 and 4 MiB) and was judged by that test's
assertions on the report file, with the native run of the same date as the reference.

| program | `WM_SUBLET=ON` | `WM_SUBLET=OFF` |
|---|---|---|
| `replay` | exit 0; `WM completed=1661 allocs=1000 checksum=15475871146958098728 system=800 peak=33`, the native line | the same line |
| `allocator-example` | exit 0, `ALLOCATOR_EXAMPLE wireshark PASS pointer_bytes=16` | the same |

Each of the trace's 20 epochs resets all four pools and then allocates from them again, so the
protected replay executes `CRENEW` on the kept blocks of `block` and `block_fast` and then derives
from the renewed generations.

The native suite of the same tree: `replay`, `source-preparation` and `allocator-example` pass.
`port-support` runs 54 tests and one errors, `test_a_rebuilt_sdk_is_invisible_at_head`, which asks
`git log` for a file's revision and ran in a copy of the tree without history.

`capstone/tests/exact-bounds-iii/wmem-q15-check.py` on the patched header: PASS at 6,000 and at
60,000 issues, no failure in any class, both controls fire.

## Platform

The tree's virtual kit on the processor model with the one-operand `CREVOKE` and `CRENEW`;
`result.json` carries the file hashes, the image hashes and the result lines. CPU properties:
`x-capstone-test-csrs=true,x-capstone-exact-bounds=true` on top of the staged runner's defaults.
The module was loaded without parameters (`unpinned=0`, `supervisor_slots=1`). One boot ran the
gates of the CPython application, the pymalloc component and the wmem component, ten processes
under `capstone-vexec`. After the last one `capstone-vexec --stats` reported 0 live domains, 0 live
regions, 0 live bytes and 0 tag pages, and the console holds no domain fault.
