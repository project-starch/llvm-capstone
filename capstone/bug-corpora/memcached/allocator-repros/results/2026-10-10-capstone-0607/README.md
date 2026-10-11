# memcached allocator-repros 06 and 07 on the Capstone `spatial` (mode 0) and `sublet` (mode 1) modes -- 2026-10-10

Pre-registered: `docs/history/10-10-2026_17-20-00_bug-corpus-audit-three-programs.md`, P4, commit `bbc7db0bd3d5`.
Until now these four cells carried oracle text only, backed by prose about a 2026-10-05 run with no result line.

    python3 capstone/bug-corpora/memcached/allocator-repros/runners/capstone-domain/run-defects.py <out> \
      --domain-build <shared/build-cases.sh capstone-domain output> --linux-build <linux-guest preset build> \
      --cases 6,7 --modes spatial,sublet [--negative-control]

with `CAPSTONE_GUEST_COMMAND_TIMEOUT=400`: the sublet-mode domain prints thousands of debug lines, and at the default 90 s
the session was cut after the domain had already exited 0 and before the runner's end marker (two such boots
read 'BOOT PRODUCED NO RESULT', exit 75 -- infrastructure, never scored).

## The cases

    case 6 spatial  completed=1 passed=True  domain 5d448c5bb04aefee  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 6 sublet   completed=1 passed=True  domain 5d448c5bb04aefee  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 7 spatial  completed=1 passed=True  domain fb50618883f5a9f0  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 7 sublet   completed=1 passed=True  domain fb50618883f5a9f0  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9

    OK   case=6 suffix-write-no-space              spatial
    OK   case=6 suffix-write-no-space              sublet
    OK   case=7 unterminated-key-read              spatial
    OK   case=7 unterminated-key-read              sublet
    4/4 arms passed

## Negative control (a corrupted fixture: every oracle must refuse)

    case 6 spatial  completed=None passed=False  domain 5d448c5bb04aefee  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 6 sublet   completed=None passed=False  domain 5d448c5bb04aefee  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 7 spatial  completed=None passed=False  domain fb50618883f5a9f0  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9
    case 7 sublet   completed=None passed=False  domain fb50618883f5a9f0  loader 2fef2a1d064ffbab  qemu 6e5e136a070b0ee9

    FIRED case=6 suffix-write-no-space              spatial
    FIRED case=6 suffix-write-no-space              sublet
    FIRED case=7 unterminated-key-read              spatial
    FIRED case=7 unterminated-key-read              sublet
    negative control: 4/4 oracles fired; 0 reported a pass on an input that never ran the case

**Result: as predicted.** Both cases COMPLETE on both modes: case 06's write lands inside the slab chunk that mode 0's
alias and mode 1's region both bound at the chunk, and case 07's read stays inside it under the fixture's 64-byte scan cap.
The negative control fired on all four arms. The Sublet carve port (`sublet-carve`) is what catches 06 and 07.
