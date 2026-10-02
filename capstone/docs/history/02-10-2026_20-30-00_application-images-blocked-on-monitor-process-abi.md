# Application-SDK images cannot run on this host: the monitor has no process ABI (2026-10-02)

**What was being done.** Running FFmpeg app fixtures 24 and 25 — two upstream defects live at the
9.0.1 pin, registered in `95aa3d340f80` — on the `level0`, `shrink` and `sublet` heap arms. The
images were built; only the guest side was missing.

**Result: blocked, with the cause located exactly.** The OpenSBI monitor on this host implements
**15 capstone SBI functions and none of the `SBI_CAPSTONE_PROCESS_*` family**, which the delegated
runtime's kernel module calls on every region creation. No source on this machine, at any pinned or
checked-out revision, contains that family.

## What was built and verified working

Three of the four pieces now exist and were each proved by a positive check, not by absence:

| piece | how it was obtained | evidence it works |
|---|---|---|
| qualified compiler | built `7d01722aab88`, Release + assertions, 19 min, 1.2 GB | `__SIZEOF_INTCAP__` and `__uintcap_t` compile; a trivial file compiles as the must-compile control |
| the three heap arms | `ports/ffmpeg/app/host/build-domain.sh` with that compiler | 13 fixture images per arm, including `ffapp_fx24.dom` and `ffapp_fx25.dom` |
| pinned kernel module | `package/modcapstone` from buildroot `8fd1ea1249a9`, built out-of-tree against the existing `linux-custom` tree | `strings` shows `parm=process_cache_bytes`; the installed module shows **zero** `parm=` lines, so the check is known to be able to fail |
| `capstone-exec`, `capstone-job` | `runtime/exec` + the pinned `libcapstone.c`, guest gcc | both link; in the guest, `insmod` succeeds, the parameter reads back `402653184`, and **`capstone-exec --stats` succeeds** |

`capstone-exec --stats` succeeding is the load-bearing one: it is the probe `capstone-vm`'s own boot
uses, and it proves the launcher and the managed module talk to each other.

## Where it stops, and why

Running an actual image fails with

    capstone-exec: cannot allocate launch regions        (exit 125)

and **no kernel message at all** — `dmesg | grep -i capstone` returns only the module-taint line.
That silence is the clue, and it is what the chain explains:

1. `exec.c:737` asks for three regions. The descriptor read out of `ffapp_fx24.dom` is healthy —
   `exchange_bytes=262144`, `contexts=15`, so META 260 KiB, DATA 4 MiB, STARTUP 64 KiB. CMA is
   fine too: 1024 MiB reserved, 1046528 kB free. Neither size nor memory is the problem.
2. `capstone.c:663-673`: once `IOCTL_PROCESS_ENABLE` succeeds on a descriptor, `owner->managed` is
   set and **every later ioctl routes to `process_ioctl`**, never to the legacy handlers. The
   legacy `ioctl_create_region` is the one that `pr_alert`s on failure — which is why nothing is
   logged.
3. `process.c:205` `process_create_region` calls
   `sbi_ecall(SBI_EXT_CAPSTONE, SBI_CAPSTONE_PROCESS_REGION_CREATE, …)` and, on error, returns
   `-ENOSPC` **silently**.
4. `SBI_CAPSTONE_PROCESS_REGION_CREATE` is `0x25` in `package/modcapstone/include/process-abi.h`,
   one of nine functions `0x21`–`0x29`.
5. The monitor built into this host's `fw_jump.elf`
   (`caplifive-buildroot/build/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.c:1810`)
   dispatches exactly these, and nothing else:

       DOM_CREATE, DOM_CALL, DOM_CALL_WITH_CAP, REGION_CREATE, REGION_SHARE, DOM_RETURN,
       REGION_QUERY, DOM_SCHEDULE, REGION_COUNT, REGION_SHARE_ANNOTATED, REGION_REVOKE,
       REGION_DE_LINEAR, REGION_POP   — 15 cases, `grep -c PROCESS` = **0**

So the kernel module and the launcher implement the process ABI; the firmware underneath does not.

## Why rebuilding does not fix it

- Buildroot `d04bd83b13cd` → the pinned `8fd1ea1249a9` is **18 files, ~1,000 lines**, confined to
  `package/modcapstone`, `package/capstone-runtime` and `scripts/`. **OpenSBI is untouched**, so
  rebuilding buildroot at the pin rebuilds the same monitor.
- `caplifive-system` is checked out at **exactly** its pinned `da972182cb24` and contains no
  `SBI_CAPSTONE_PROCESS*` either.
- Neither QEMU revision — the checked-out `deb7d75756` nor the pinned `674cdab03c3e` — mentions it;
  this extension is the monitor's, not the emulator's.

A repository-wide search for `SBI_CAPSTONE_PROCESS_REGION_CREATE` outside the modcapstone header
returns nothing. **The implementation is not on this machine in any revision.**

## What this means

`dev`'s delegated runtime expects a monitor ABI that none of the submodule pins provide. Either the
monitor work lives on a branch never pushed or never fetched here, or the pins on `dev` are
inconsistent. That is a question for whoever owns the runtime, not something a port lane can settle.

**Consequences for the port work:**

- No application-SDK image can run on this host: FFmpeg and tshark app fixtures, and the heap arms
  generally. Fixtures 24 and 25 stay built and registered; their predictions are untouched.
- What **does** run here is unaffected: bare `domain_main` images through
  `tests/runtime-qemu/run-domain-smoke.py` and the port's own `domain-loader`. That is the path the
  bug corpora use, and the memcached corpus ran 10/10 on it in this cycle.
- A plain-`malloc` corpus case is therefore **not** reachable either: the runtime heap
  (`level0.c`, `sublet_heap.c`) is linked only by `capstone_configure_application`, which produces
  an application image. Bare-domain corpus cases bring the port's own allocator instead.

## Reproducing the diagnosis

    # the monitor's whole capstone SBI surface
    grep -A60 'case SBI_EXT_CAPSTONE:' \
      caplifive-buildroot/build/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.c \
      | grep 'case SBI_EXT_CAPSTONE'

    # what the module asks for
    grep -n 'SBI_CAPSTONE_PROCESS' <pinned>/package/modcapstone/include/process-abi.h

The staged, working pieces are kept at `/tmp/capstone/pinned-platform/`: the built module, the two
guest binaries, and the share that reproduces `EXEC_STATS_OK` followed by the region failure.

## One instrument note

Raising the guest console's log level (`dmesg -n 8`) floods the serial line with
`remote fence extension is not available in SBI v1.0` and the reader times out — a run that looks
like a guest stall but is a drowned console. Capture to a file on the 9p share instead; that is how
the module's silence was established rather than assumed.
