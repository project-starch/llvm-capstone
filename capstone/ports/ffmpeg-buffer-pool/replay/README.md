# Replay: native control and Capstone domains

This directory builds the same pool replay engine for two execution targets.
It also contains the Linux loader used to enter a Capstone domain.

| File | Where it runs / what it produces |
|---|---|
| `replay.c` | Native executable on the development machine, or a Capstone domain selected by `FFPOOL_DOMAIN` |
| `host.c` | Linux process inside QEMU; loads the domain, grants its memory regions and retrieves results |
| `port.py` | Development machine; applies our allocator changes to instrumented FFmpeg sources |
| `build.sh native` | Development machine; creates `combined-port-native/replay` and `security` |
| `build.sh capstone` | Development machine; creates `combined-port-capstone/replay.dom`, `security.dom` and `host.user` |
| `run-qemu.sh` | Development machine; boots QEMU and invokes `host.user` in guest Linux |

Output paths are relative to `$FFPOOL_WORK`. Both builds link the allocator
code from `../runtime/` and the observer from `../trace/`.

## Selecting protection

`run-qemu.sh TRACE RESULT-DIR MODE` selects `0` (Capstone bounds), `1` (also
backing lifetime), or `2` (also Sublet pool-lease lifetime). All three modes
execute the same `replay.dom`; there is no separate Sublet binary.

The native replay accepts `TRACE OUTPUT [MODE]` with the same numbers, but its
ordinary pointers do not enforce capability bounds or revocation. It checks
functional agreement using the ported allocator layout. This native replay
is distinct from the full upstream decoder built by `../record/`.

See the [complete workflow](../README.md#build-and-run) for setup and commands.
