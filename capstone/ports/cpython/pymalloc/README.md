# CPython 3.13.7 pymalloc

The real allocator in `Objects/obmalloc.c`, extracted into a standalone replay. This is an
allocator component port with a native Python workload recorder; it does not run the
interpreter (`ports/cpython/app` does). The free-threaded/mimalloc configuration is outside scope.

## Targets

| preset | what it builds |
|---|---|
| `capstone-application` | the replay as a Capstone process on the virtual profile (`CAPSTONE_SDK`); `-DPYMALLOC_SUBLET=ON` adds the Sublet protection |
| `cheribsd` | the replay for stock CheriBSD purecap ([host/cheribsd](host/cheribsd/README.md)) |
| `native` | the replay, the unpatched `replay-reference`, the recorder and the tests |

Every target also builds `allocator-example` (`examples/pymalloc.c`), the link example the shared
CheriBSD harness runs (`ports/common/host/cheribsd/components.json`).

## Patches

`upstream.json` pins the official CPython 3.13.7 archive by SHA256. `cmake/prepare-source.py`
applies the ordered patches by variant:

| patch | what it does | `reference` | `ported` | `protected` |
|---|---|:-:|:-:|:-:|
| 0001 standalone-pymalloc | extracts the allocator from the interpreter | x | x | x |
| 0002 capability-provenance | real pool sentinels; arena and pool pointers from retained authority (`backing.c`), never rebuilt from integers | | x | x |
| 0003 blocks-as-sublet-lifetimes | the Sublet protection: every block a child lifetime of its arena (`CDERIVE`), revoked on free (`CREVOKE`) | | | x |

`PYMALLOC_SUBLET=ON` (capstone-application only) selects `protected`; the native
`replay-reference` is `reference`. The interpreter carries the same protection as its patch 0014.

The pinned 64-bit configuration uses 16 KiB pools, 1 MiB arenas, 16-byte size classes and a
512-byte small-request threshold. Capability pointers enlarge the pool header from 48 to 80
bytes; both retain the upstream computed `POOL_OVERHEAD`.

`src/shared/backing.c` gives the replay its arenas from one 64 MiB region and a 16 MiB metadata
heap for the replay's own tables and for requests over 512 bytes; it is the replay's stand-in
for the interpreter's system allocator, not a port of libc malloc.

## Build

From the repository root, source `capstone/tests/capstone-test-env.sh`, then from this directory:

```sh
cmake --preset native && cmake --build --preset native && ctest --preset native
cmake --preset capstone-application -DCAPSTONE_SDK=<virtual SDK> [-DPYMALLOC_SUBLET=ON]
cmake --build --preset capstone-application
```

Presets build under `/tmp/capstone/cpython-pymalloc/build/`; sources and outputs stay outside
the repository.

## Record and replay

The optional recorder needs a native, GIL-enabled, little-endian CPython 3.13.7 with development
headers:

```sh
cmake --preset native -DPYMALLOC_RECORD_PYTHON=/path/to/python3.13
cmake --build --preset native
PYTHONMALLOC=pymalloc /path/to/python3.13 host/record.py /tmp/pymalloc-stdlib.bin \
  --module-dir /tmp/capstone/cpython-pymalloc/build/native/python --rounds 20
/tmp/capstone/cpython-pymalloc/build/native/bin/replay /tmp/pymalloc-stdlib.bin /tmp/pymalloc-native.bin
```

The recorder wraps both MEM and OBJ APIs in the actual interpreter and records successful
requests in a bounded, single-interpreter workload (JSON parsing, regex matching, bytearray
allocation and resizing). Replay starts from an empty allocator and checks synthetic payloads at
every free, realloc and END; it replays API requests, not Python object graphs. The native tests
compare the replay against `replay-reference`: payloads, counters, and an allocation-decision hash
normalized by arena identity and offset.

## History

Until 2026-10-10 this port also carried a lifetime adapter behind a hook interface (patch 0003
`block-lifetime-hooks`, `src/allocators/sublet/`), a freestanding Capstone domain target, a
linux-guest loader and a CheriBSD PoisonCap backend with its own README and runner. The
CDERIVE/CREVOKE instructions made the adapter unnecessary, and the other targets are not part of
the virtual platform. Their recorded results stay in `results/`.
