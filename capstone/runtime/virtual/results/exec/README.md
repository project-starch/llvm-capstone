# run-exec.py: the write-denial gate, 25 of 25

[`result.json`](result.json) is `run-exec.py`'s own record of one run, 2026-10-21, on QEMU's virtual
profile, one hart: status `PASS`, 25 of 25 checks (the 24 of the first version and
`denial_check_rejects_other_errors`, the negative control for the denial checks). Every check in the
file is true; `lines` holds the guest's `@@` lines each check reads, and `sha256` the inputs.

What it was run on:

- **Tree**: `ac1c28b76093`, this branch on the #250 head (`6024f00e4f2f`). `source_sha256` hashes the
  gate's own sources. The two control arms below ran from the exported tree of this branch's previous
  commit (`4cdbd9e06dc2`); the contract and runner sources of the two trees are byte-identical
  (compared by hash), so the same programs ran in every arm.
- **Platform**: QEMU `bd6cd022b077` (`svadu=false`, `x-capstone-supervisor-slots=32`), the Image,
  `fw_jump.elf` and `rootfs.ext2` hashed in `result.json`, launcher `e4a9669bc9a6`, module
  `65ad4a0c4010` (flat table). The module is this tree's `capstone_vm.c` built against kernel 6.1.0 with
  the supervisor-slot default changed from 1 to 32 in the build copy only, which is the gate platform's
  setting; `exec` creates no threads, so it does not depend on it. The same module with the default
  left at 1 fails `run-pthreads.py` at its first `pthread_create`, as the #250 head's module does with
  the same default.
- **Application**: `exec-contract.c` built with the SDK of this tree and the SPEC toolchain v4 clang
  (`toolchain-record.txt` in that build; an integration build of the compiler, not dev alone).
- **Helper**: `memfd-launch.c`, static, built with the QEMU buildroot's riscv64 cross compiler.

The two checks the reviewer asked to tighten read "Text file busy" from the guest shell and fail on
any other error. In this run the three denial lines are `denied ...: can't create /tmp/e.dom: Text
file busy`, the same for `/mnt/e4/e.dom` and for the hard link, and the control, a write to
`/proc/version`, reads `denied sh: write error: Input/output error` and is rejected by `busy_denial()`.

## The other gates, against controls

Same SDK, compiler, contracts and runners for every arm (this tree's); only the platform differs: this
branch, the #250 head, and dev `4985df88`. A comparison of failing sets, which is blind inside a check
that fails on both sides.

| gate | dev | #250 head | this branch |
|---|---|---|---|
| `run-hosted.py` | 0 of 76 failing | 0 of 76 | 0 of 76 |
| `run-pthreads.py` | 0 of 10 | 0 of 10 | 0 of 10 |
| `run-nodes.py` | 10 of 16 | same 10 | same 10 |
| `run-malloc.py` | 6 of 23 | same 6 | same 6 |
| `run.py` | 10 of 47 | 8 of 47 | same 8 as the #250 head |

`run.py` differs from dev by `vm-mappings` and `vm_mappings_contract`, which #250 makes pass, and not
between the #250 head and this branch. The nodes, malloc and runtime failures are the same checks in all
three arms, so this change neither adds nor removes any; what they fail on was not looked at here.
Not run: `run-pin-ahead.py`, `run-fp.py`, `run-step-hooks.py`, the memcached `run-virtual.py`.
