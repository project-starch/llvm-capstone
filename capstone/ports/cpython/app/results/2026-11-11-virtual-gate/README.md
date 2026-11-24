# CPython application gate on the virtual profile, 2026-11-11

The cpython row of `capstone/runtime/virtual/run-ports.py` (the same command, library staging and
oracle), run for both builds of the port after patch 0014 and the processor model changed their
lifetime instructions. `run-ports.py` itself needs the PostgreSQL, FFmpeg and tshark images of the
same invocation, so the row ran from a stage of its own.

| image | build | exit | `CPYTHON-OK 1000` |
|---|---|---|---|
| nested, `00c525b78717` | `CPY_SUBLET=1 build-virtual.sh cpython` | 0 | yes |
| stock, `83fa4a073d33` | `build-virtual.sh cpython` | 0 | yes |

The workload is `import json,gc`, a round trip of 1,000 dictionaries through `json`, and a
collection, with `PYTHONHOME` on the staged library compiled by the build's native 3.13.7.

The nested image carries 12 `CDERIVE` and 5 one-operand `CREVOKE` (`_PyObject_Free`,
`_PyObject_Realloc`, and mallocng's `free` and `realloc`), the stock image 6 and 3.

## Platform

The tree's virtual kit on the processor model with the one-operand `CREVOKE` and `CRENEW`;
`result.json` carries the file hashes, the image hashes and the result lines. CPU properties:
`x-capstone-test-csrs=true,x-capstone-exact-bounds=true` on top of the staged runner's defaults.
The module was loaded without parameters (`unpinned=0`, `supervisor_slots=1`). One boot ran the
gates of the CPython application, the pymalloc component and the wmem component, ten processes
under `capstone-vexec`. After the last one `capstone-vexec --stats` reported 0 live domains, 0 live
regions, 0 live bytes and 0 tag pages, and the console holds no domain fault.
