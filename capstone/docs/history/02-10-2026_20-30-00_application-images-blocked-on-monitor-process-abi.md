# Application-SDK images cannot run on this host: the monitor has no process ABI (2026-10-02)

> ## CORRECTED 2026-10-03 — the central argument of this note was wrong
>
> Application images **do** run on this host, and the results are in
> [`ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/`](../../ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/README.md).
> Four claims below are withdrawn; the sections themselves are left standing with their errors
> marked, because the reasoning is what went wrong and hiding it teaches nothing.
>
> 1. **"The implementation is not on this machine in any revision" is FALSE**, and was false when
>    written. The monitor with the process ABI was already built at
>    `/tmp/capstone/deleg-gate2/opensbi-T/` (source dated 2026-09-30, built 2026-10-01 10:18).
> 2. **"OpenSBI is untouched, so rebuilding buildroot at the pin rebuilds the same monitor" is
>    FALSE**, and this is the error that mattered. The pin diff *does* bump `components/opensbi`
>    (`49b0f932` → `cf344cf3`), which carries a nested bump of `lib/sbi/capstone-sbi`
>    (`2c49c41c` → `4674ab6a`), and the pinned revision **has** the process ABI. Rebuilding
>    buildroot at its own pin would have fixed this. I read a truncated `--stat` tail and never
>    saw the five non-package paths the diff also touches.
> 3. **"Either the monitor work is on an unfetched branch, or the pins on `dev` are inconsistent"
>    is FALSE on both horns.** The objects resolve locally and **the pins are consistent**: the
>    QEMU used is a checkout at exactly the pinned `674cdab03c3e`, and the monitor's
>    `capstone-sbi` is byte-identical to the pinned `4674ab6a`. What lags the pins is the *main
>    clone's working trees* — buildroot at `d04bd83b13cd`, QEMU at `deb7d757`. This note searched
>    checkouts and reported the result as a statement about pinned revisions.
> 4. **Two counts are wrong**: the installed monitor dispatches **14** capstone SBI functions, not
>    15 (`REGION_SHARE_CHILD` at :1873 fell outside the `-A60` window this note's own command
>    used), and the process family is **ten** functions `0x21`–`0x2b`, not nine `0x21`–`0x29`.
>
> What stands: the **installed** platform has none of the process ABI (`grep -c PROCESS` on its
> `sbi_capstone.c` is 0, and `context_step` is 0 in its `fw_jump.elf` against 41 in the pinned
> one), which is why nothing ran until the pinned pieces were used.

**What was being done.** Running FFmpeg app fixtures 24 and 25 — two upstream defects live at the
9.0.1 pin, registered in `95aa3d340f80` — on the `level0`, `shrink` and `sublet` heap arms. The
images were built; only the guest side was missing.

**Result: blocked, with the cause located exactly.** The *installed* OpenSBI monitor implements
**14 capstone SBI functions and none of the `SBI_CAPSTONE_PROCESS_*` family**, which the delegated
runtime's kernel module calls on every region creation. ~~No source on this machine, at any pinned
or checked-out revision, contains that family.~~ **WITHDRAWN** — see the correction above: the
pinned revision has it, and it was already built on this host.

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
   one of **ten** functions `0x21`–`0x2b` (corrected: this note first said nine, `0x21`–`0x29`).
5. The monitor built into this host's `fw_jump.elf`
   (`caplifive-buildroot/build/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.c:1810`)
   dispatches exactly these, and nothing else:

       DOM_CREATE, DOM_CALL, DOM_CALL_WITH_CAP, REGION_CREATE, REGION_SHARE, DOM_RETURN,
       REGION_QUERY, DOM_SCHEDULE, REGION_COUNT, REGION_SHARE_ANNOTATED, REGION_REVOKE,
       REGION_DE_LINEAR, REGION_POP   — 15 cases, `grep -c PROCESS` = **0**

So the kernel module and the launcher implement the process ABI; the firmware underneath does not.

## ~~Why rebuilding does not fix it~~ — WITHDRAWN, the premise is false

**Every bullet below is wrong about the buildroot pin.** It bumps `components/opensbi`, whose
nested `capstone-sbi` gains the process ABI, so rebuilding at the pin *would* have fixed this.
Left in place so the mistake is legible: the `--stat` output was read truncated.

- Buildroot `d04bd83b13cd` → the pinned `8fd1ea1249a9` is **18 files, ~1,000 lines**, confined to
  `package/modcapstone`, `package/capstone-runtime` and `scripts/`. **OpenSBI is untouched**, so
  rebuilding buildroot at the pin rebuilds the same monitor.
- `caplifive-system` is checked out at **exactly** its pinned `da972182cb24` and contains no
  `SBI_CAPSTONE_PROCESS*` either.
- Neither QEMU revision — the checked-out `deb7d75756` nor the pinned `674cdab03c3e` — mentions it;
  this extension is the monitor's, not the emulator's.

~~A repository-wide search for `SBI_CAPSTONE_PROCESS_REGION_CREATE` outside the modcapstone header
returns nothing. **The implementation is not on this machine in any revision.**~~ **WITHDRAWN.**
The search covered working trees, not pinned revisions, and not `/tmp/capstone/deleg-gate2/`.

## What this means

`dev`'s delegated runtime expects a monitor ABI that none of the submodule pins provide. Either the
monitor work lives on a branch never pushed or never fetched here, or the pins on `dev` are
inconsistent. That is a question for whoever owns the runtime, not something a port lane can settle.

**Consequences for the port work:**

- ~~No application-SDK image can run on this host~~ **WITHDRAWN**: fixtures 24 and 25 ran on all
  three heap arms on 2026-10-03 once the pinned monitor and QEMU were used.
- What **does** run here is unaffected: bare `domain_main` images through
  `tests/runtime-qemu/run-domain-smoke.py` and the port's own `domain-loader`. That is the path the
  bug corpora use, and the memcached corpus ran 10/10 on it in this cycle.
- The structural half still stands: the runtime heap (`level0.c`, `sublet_heap.c`) is linked only
  by `capstone_configure_application`, so a plain-`malloc` case must be an application image and
  cannot be a bare-domain corpus case. ~~therefore not reachable~~ — it *is* reachable, through
  the application fixtures, which is what fixtures 24 and 25 are.

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
