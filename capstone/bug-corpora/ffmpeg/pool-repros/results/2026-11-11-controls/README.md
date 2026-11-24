# FFmpeg nested-lifetime controls, 2026-11-11: parent, sibling, successor, stale

The controls of [`../2026-10-11-controls`](../2026-10-11-controls/README.md), run again after the
buffer-pool port's patch 0003 changed its design. The pre-registration is unchanged:
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md).
The protected arm is patch 0003 on 0001 (`FFPOOL_SUBLET=ON`), the release arm the same source
without 0003. Sources: [`controls/nested-lifetime/`](../../controls/nested-lifetime/README.md).

**All 12 predictions held**: 6 per boot, one boot per arm. The stale read completed on release,
reading C's data, and faulted on protected with cause 25 at the labelled probe. The 2026-10-11
record ran three boots per arm.

## What changed under the controls

Patch 0003 no longer keeps `sublet_life` and `sublet_issued` in `BufferPoolEntry` and derives no
child per hand-out. `pool_release_buffer` renews the entry's own lifetime,
`entry->data = CRENEW(entry->data)`. After the release `entry->data` names the entry's new lifetime
and the user's copy names the one that ended. Control 91 reads and writes through `entry->data`
only; the 2026-10-11 run also went through `sublet_life`, which no longer exists.

## Outcomes

| control | arm | outcome |
|---|---|---|
| 91 parent | release | completed: the parent read returns A's 0x5A with 0 mismatches |
| 91 parent | protected | completed: the same |
| 92 sibling | release | completed: B intact (0 mismatches against 0xB2) |
| 92 sibling | protected | completed: B intact |
| 93 successor `run` | release | completed: C == A after 1 get (`0x3f7f747440`) |
| 93 successor `run` | protected | completed: C == A after 1 get (`0x3fbf053440`) |
| 93 `stale` | release | **completed**: the read through A's old pointer returned C's 0x77 |
| 93 `stale` | protected | **fault, cause 25**, at `stale_read_probe`+32, `93_ctl_successor_entry/case.c:36` (link pc `0x1dea4`); fault address `0x3f80827440` = A = C |

Platform controls, both boots: `bounds-malloc` faulted with cause 28 and `uaf-malloc` with cause 25.

## Identity

`results.json` carries each boot's record and, under `revision`, the tree's revision line, the
hash of patch 0003 and of every control file.

| what | value |
|---|---|
| QEMU / launcher / module | `2d5a7107...` / `0f8e0f90...` / `e47bca62...` |
| options | `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, `-smp 1`, module loaded without parameters |
| SDK | the corpus SDK (`-fno-builtin`), virtual profile, mallocng |
| images, protected | `ctl-91` `b46ef27a...`, `ctl-92` `e31dc291...`, `ctl-93` `8d2d458c...` |
| images, release | `ctl-91` `6368203f...`, `ctl-92` `b695df80...`, `ctl-93` `9fbff683...` |

## Deviation

`run-controls.py` resolves a fault's source line with `llvm-addr2line`, which the toolchain
directory of this run does not ship. The run passed `--llvm-bin` a directory of links to that
toolchain's tools plus `llvm-addr2line` as a link to its `llvm-symbolizer`. A first pair of boots
without it booted and then stopped in the runner before any record was written.
