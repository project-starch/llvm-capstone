# SQLite 3.22.0 with a nested heap on the one-operand lifetime instructions (2026-11-11)

The application gate of this port, run on the three builds that carry memsys5, after the lifetime instructions
changed (`CREVOKE` takes the object alone; `CRENEW` ends an object's lifetime and begins a new one in its place):

- **stock**: `SQLITE_HEAP=memsys5`, memsys5 as SQLite ships it;
- **protected**: `SQLITE_HEAP=memsys5-sublet`, `../../../sublet/sublet-3220000-memsys5.patch`;
- **block port**: `SQLITE_HEAP=memsys5-sublet-block`, `../../../sublet/sublet-3220000-memsys5-block.patch`. This value
  of `SQLITE_HEAP` is new in `build-domain.sh`. The block port is the only SQLite code that executes `CRENEW`: a realloc
  that keeps its memsys5 block (`memsys5Renew`) or its lookaside slot (`lookasideRenew`) renews the lifetime.

Two workloads: `run-app.py` as the port's README describes it, which leaves memsys5 unconfigured, so SQLite allocates
from the C library and only the lookaside half of the patches runs; and speedtest1 with `--heap`, which configures
memsys5, as in [`../2026-10-11-same-answers/`](../2026-10-11-same-answers/README.md). The full record is `results.json`.

## Outcome

| check | stock | protected | block port |
|---|---|---|---|
| `run-app.py` `work` | **FAIL**, one line (see the next section) | the same FAIL | the same FAIL |
| `run-app.py` `again` | PASS | PASS | PASS |
| `run-app.py` `speedtest1` (`--size 1`, default allocator) | PASS, 32 phases equal to native | PASS, 32 phases | PASS, 32 phases |
| speedtest1 `--heap 8388608 64 --size 1` | exit 0, 32 phases equal to native | exit 0, 32 phases equal to native | exit 0, 32 phases equal to native |
| speedtest1 `--heap 1048576 64 --size 1` | exit 1, 13 phases equal to native, then "out of memory" at `VACUUM` | the same | the same |

No arm logs a domain fault in any of the five runs. The two `--heap` rows are the 2026-10-11 result for stock and
protected (32 phases at 8 MiB; out of memory after 13 phases at 1 MiB, a capacity limit of the configuration), and the
block port reads the same.

## The one failing check, and why it is not the patches

`work` compares the shell's output with the native shell's, byte for byte. The guest's output has one line the native
output lacks, and is otherwise identical (28 lines against 27):

    capstone-exec: exit image=/mnt/host/sqlite3-<arm>.dom sha256=...

It is the launcher's own diagnostic for the second writer, the process the shell starts through `.system` while it
holds `BEGIN IMMEDIATE`. `run-app.py` exports `CAPSTONE_EXEC_DIAGNOSTICS=1` to see a domain fault's record, the
second writer inherits it, and `second-writer.sh` folds the child's stderr into the shell's output. The second writer
itself is refused as required (`second writer exit 5`). The line is there on the stock arm, which has no patch, with
the same position. The check is recorded as it read; neither the runner nor the expectation was changed.

## Platform: this gate needs more than one context

`work` runs two Capstone processes at once. The corpus kit's module has one context (`supervisor_slots=1`, a compiled
default), and on it the second writer waits for the context the shell holds while the shell waits for the second
writer: the first attempt of this gate stood still there for 14 minutes and was stopped. The run recorded here uses the
landing gates' copy of the same module with 32 contexts and `x-capstone-supervisor-slots=32`, on the same QEMU, kernel,
firmware and launcher.

| file | sha256 |
|---|---|
| qemu-system-riscv64 | `2d5a7107890e48475fada73439d90ffd5c7cff166eb1d547d7a9fc5f31722365` |
| capstone-vexec | `0f8e0f903725d734817b3e30d79127c290f3480a641d4f6de8bc7f699d05c76e` |
| capstone_vm.ko (32 contexts) | `38d4d411fb19f2c23cb4398fddd7534828a5c0468e7d20cd16a49d94ce2efcd6` |
| Image | `48eb725a40e448fa9ac85abb8b7f9fe8a4080d73ffd49c051bed111319d64da2` |
| fw_jump.elf | `5b5c047a00306f35c99bc4b506ec840c784fe1b73dd13ea6ecb4cf793fece9af` |
| rootfs.ext2 | `62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845` |

One persistent VM (`capstone_vm up --profile virtual --exact-bounds --cpu-prop x-capstone-test-csrs=true --cpu-prop
x-capstone-supervisor-slots=32`), the module loaded without parameters (pinned), 8 GiB.

## Builds

| | stock | protected | block port |
|---|---|---|---|
| `src/sqlite3-capstone.c` | `7fc92e19a3af61f3...` (the prepared `sqlite3.c`) | `f05b67f965d03bf3...` | `81f1aec827b93cd3...` |
| `sqlite3.dom` | `e9b3ec39512cbb4e...` | `225a8a368c11f223...` | `7a7822e35f2ed144...` |
| `speedtest1.dom` | `8e242be6899e022a...` | `0616616d88c95425...` | `4e7dec1f2d3172a6...` |
| `CDERIVE`, `CREVOKE`, `CRENEW` words | 6, 3, 0 | 16, 8, 0 | 16, 8, 22 |

The stock image's 6 and 3 are the C library's mallocng. Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at
ea57497908. SDK: `runtime/virtual/build-sdk.sh` without the corpus gate; `-O2`. Patches:
`sublet-3220000-memsys5.patch` f96ee0329a08..., `sublet-3220000-memsys5-block.patch` 508f9f53b0de....

## Commands

```sh
SQLITE_322_CACHE_ROOT=<cache> bash capstone/ports/sqlite/app/prepare-sources.sh <src>
bash capstone/ports/sqlite/app/build-native.sh <src> <native>
for h in memsys5 memsys5-sublet memsys5-sublet-block; do
  SQLITE_HEAP=$h bash capstone/ports/sqlite/app/build-domain.sh <sdk> <src> <out>/$h
  python3 capstone/ports/sqlite/app/run-app.py --state <vm> --share <share> --build <out>/$h --native <native> \
    --label $h --size 1 --report app-$h.json
  # in the guest, after run-app.py staged the images in the share:
  cd /tmp && rm -f st.db st.db-journal && /mnt/host/speedtest1-$h.dom --heap 8388608 64 --size 1 st.db
  cd /tmp && rm -f st.db st.db-journal && /mnt/host/speedtest1-$h.dom --heap 1048576 64 --size 1 st.db
done
```

## Limits

N = 1 per cell. The wall times in `results.json` are host times around each call, with other sessions of the same
campaign on the host, and are not a performance comparison. Nothing counts how often `CRENEW` executes: that the
block port's renewal works is read from the corpus's realloc controls
(`bug-corpora/sqlite/engine-repros/results/2026-11-11-block-port`), and here from 32 phases that give native's answers.
