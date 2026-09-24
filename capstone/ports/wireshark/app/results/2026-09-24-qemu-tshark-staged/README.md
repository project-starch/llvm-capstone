# tshark in a Capstone domain on QEMU: M0, the M1–M5 stages and the oracle (2026-09-24)

**Question.** Does the minimal tshark (Wireshark 4.6.8, `dissector-whitelist.txt`) link as one
Capstone domain, reach each stage, and print output byte-identical to native tshark? The
predictions were written and pushed to `dev` before the images linked or booted
(`predictions.txt`, commits `5397f7a` … `eee5b43`). Each prediction is scored below, refuted
ones included.

## Verdict

**On QEMU, yes.** The minimal tshark links as one 66.3 MiB Capstone domain and runs in a 128 MiB
CMA block. It reaches all five stages. On dhcp, dns_port, http and arp, on their byte-flipped
copies, and on dns-ooo, its `-V -n` output is **byte-identical to native stock tshark's stdout**,
and its stderr is byte-identical to the native minimal build's. ntp.pcap differs from stock only
where NTP is not whitelisted, and equals the native minimal build. That is the negative control
firing as it must.

Three things qualify it:
- **It took three runtime fixes the plan did not foresee.** A domain never ran `.init_array` and
  could not run `.fini_array`, and musl-capstone's `pthread_cond_t` is too small for its own
  fields. All three are fixed port-locally below, and none is tshark's own.
- **One ntp section stalled** in the guest before its domain started, and was stopped. Two later
  ntp runs returned. At first I wrote that the domain had spun; that was wrong and is corrected
  under "Stalls".
- **It is a QEMU compatibility result:** gp is fabricated, and the heap has no bounds or
  revocation (last section).

## What it took, beyond the pre-registered plan

The first boots found three runtime gaps, filed as **ISSUES C-64, C-65 and I-11**, each with a
reproducer in `tests/runtime-gaps/`. None of the three is tshark's own. A survey of 219 recent
domain images under `/tmp/capstone` (FFmpeg app, SQLite, the R1 campaigns, …) found none with a
non-empty `.init_array`/`.fini_array` or any `pthread_cond` symbol. The same check fires on tshark.
So tshark is the first domain these gaps reach, and no shipped result depends on them; the CPython
domain's image was not on disk to check. Each fix below is port-local and is shown to fire on a
stand-in. The complete fixes belong in shared infrastructure.

| boot | what happened | cause | fix |
|---|---|---|---|
| stages 1 (`-bbad`) | M1 halted in `exit()`: cause 24, a load through the integer `0x…19c4630` | musl's `libc_exit_fini` walks `.fini_array` through `uintptr_t` and loads each slot through an **integer address**: that load is the fault. There is a second defect behind it, never reached: the one slot (libxml2's `xmlDestructor`) holds the function's **link** address as a plain integer. A static domain has no relocations, and the capability initialisers do not cover these arrays (an audit scanned all 253 initialiser bodies; none writes there). The two `.init_array` constructors (GLib's `glib_init_ctor`, libgpg-error's `gpg_err_init`) had never run at all; `my_first_domain/link.ld` says "nothing in a domain calls the INIT array … for the day one appears" | `src/tsapp-init-fini.c`. It runs the constructors before `main` (`deps/domain_entry.c`, through a *defined* weak default, C-56-safe). Its `__libc_exit_fini` replaces musl's weak one. Each slot is made callable as an anchor function's code capability moved by (slot − the anchor's link address). Matched pair on a stand-in with two constructors and two destructors: output byte-identical to native (order included) with the file, and without it the same cause-24 halt on the fini slots |
| stages 2 (`-yOUM`) | M1 REACHED; M2 halted in `epan_init`: cause 24 in musl's `__private_cond_signal`, a broadcast (n = −1), on `cincoffsetimm a4, a0, 0x20` with a0 = 1 | musl-capstone's `pthread_cond_t` is 12 ints, 48 bytes: room for three 16-byte pointers. musl's internal macros put `_c_tail` at `__u.__p[5]`, 32 bytes past the object (read at +80), and `_c_shared`/`_c_head` overlap its int fields (`src/internal/pthread_impl.h`). `pthread_cond_init` zeroes 48 bytes, so +80 is whatever the neighbouring heap block holds, here 1. The broadcast came from GLib's `gthread-posix.c`, the only `pthread_cond_*` user. Which GLib function issued it is not established: `g_once_init_leave()` broadcasts on every call, but the halting image was not kept | `deps/patches/glib-0008`: in a domain (one thread) GCond signal/broadcast have no waiter to wake and are no-ops; a wait aborts. GLib's `gthread-posix.c` is the only `pthread_cond_*` user in the image |
| stages 3 (`-kCjg`) | all five REACHED and MATCH; M5 reported **no** unserved syscalls, M4 fifteen | the runtime prints its unserved list to fd 1 after the program ends, and the full tshark run leaves fd 1 closed, so the line is lost without a trace (shown on a stand-in that calls `close(1)`) | `src/tsapp-heap.c`, the exit hook, reports the list itself on fd 2 with `stdout=open\|closed` |

Found at the link, before any boot:
- **GLib's LeakSanitizer hooks** were two undefined weak symbols, whose address is not NULL in a
  domain (C-56's open half). Fixed by `glib-0007`, and the GLib recipe now refuses any undefined
  weak symbol.
- **The staged objects had lost the code after their stop.** `exit()` is noreturn, so M1's object
  came out 7.7 KB against 207 KB. Patch 0006's stop now sits behind a volatile flag, and all four
  staged objects are within 1.2% of M5's.

An adversarial audit of the runner, run before any tshark boot, found that REACHED could be
wrong: tshark exits 3 on a failed `cf_open`, the same number as stage 3. Stages now exit with
100 + n and must also print their stage line. The audit's other findings are fixed too
(`predictions.txt`).

## Results

### M0 (`host/build-domain.sh`)

- **Links:** all five images link. No undefined symbol and no undefined weak symbol.
- **Negative control:** without `hostcall.o`, exactly `__capstone_hostcall` and `domain_main` are
  undefined.
- **Constructors:** no orphaned constructor section (`.init_array.*`, `.ctors`).
- **Size:** `code_len` 69,527,424 bytes (66.3 MiB, M5; M1–M4 within 320 bytes).
  - `.bss` 40.6 MiB, almost all of it the 40 MiB level0 arena;
  - `.text` 17.1 MiB, of which 10.3 MiB is the 255 capability-initialiser functions;
  - `.data` 4.5 MiB;
  - `.rodata` 4.1 MiB.
- **Block:** 128 MiB (order 15) from CMA for every run. dmesg shows `tot_size = 8000000`, with no
  doubling.

### M1–M5 on dhcp.pcap, final images (`qemu-stages-20260924-223202-gu4p.log`)

| stage | status | output | level0 peak_end | unserved syscalls |
|---|---|---|---|---|
| M1 `main` | 101 + `TSAPP-STAGE 1` | MATCH (nothing else) | 33,744 B | none |
| M2 `epan_init` done | 102 | MATCH (the minimal build's 194 stderr lines, then the stage line) | 17,525,360 B (16.7 MiB) | uname ×3, getrandom ×2, rt_sigaction, getcwd |
| M3 capture opened | 103 | MATCH | 25,979,568 B | uname ×5, getrandom ×4, rt_sigaction, getcwd |
| M4 first frame | 104 | MATCH: stock's frame 1 exactly, 3,774 bytes | 28,121,856 B | as M3, plus rt_sigaction ×3 and rt_sigprocmask |
| M5 all frames | 0 | MATCH: stock's stdout exactly, and the minimal build's stderr | 28,123,936 B (26.8 MiB) | as M4; fd 1 closed at exit |

The same boot: 0 `EXT4-fs error`, `MODULE-MD5 5696d0fe…` (the a74a856 module), `cma=1536M`,
image hashes as in `SHA256SUMS`.

### The oracle: M5 on the captures

Counted boots: `-Amjk` (http, arp and their flips) and `-lFvW` (dhcp, dns_port and their flips,
dns-ooo, ntp). Every run used the final M5 image (`8d294cc3…`). Neither boot logged an EXT4 error,
and both used the a74a856 module.

| capture | stdout vs stock | stderr vs minimal | status | flip control |
|---|---|---|---|---|
| dhcp | MATCH | MATCH | 0 | — |
| dhcp.flip | MATCH | MATCH | 0 | FIRES (stock changed 2 lines) |
| dns_port | MATCH | MATCH | 0 | — |
| dns_port.flip | MATCH | MATCH | 0 | FIRES (2 lines) |
| http | MATCH | MATCH | 0 | — |
| http.flip | MATCH | MATCH | 0 | FIRES (2 lines) |
| arp | MATCH | MATCH | 0 | — |
| arp.flip | MATCH | MATCH | 0 | FIRES (2 lines) |
| dns-ooo | MATCH | MATCH | 0 | — |
| ntp | **DIFFERS** (24 lines: stock's NTP tree against the minimal build's `Data`) | MATCH | 0 | — |

ntp's domain stdout equals the native minimal build's byte for byte, in this boot and in the
ntp stages boot (`-CVwR`).

An audit regenerated all ten references natively from the capture bytes recorded in each boot's
`.sha256` sidecar. It found them identical to the stored ones, and re-split every raw output with
the same verdicts. In each flip pair the domain's two outputs differ by exactly the flipped field
(for example `Keep-Alive` → `Keep-Amive`, or one address octet), so the domain dissected the bytes it
was given. What "stderr MATCH" can show: the domain's fd 1 and fd 2 arrive merged, so it means the
merged stream begins with the minimal build's stderr, not which descriptor carried it.

Every run's unserved syscalls are the same:
- uname ×5, getrandom ×4, rt_sigaction ×4, getcwd and rt_sigprocmask;
- fd 1 is closed at exit.

### Predictions, scored

| | prediction | outcome |
|---|---|---|
| P1.1 | all five link; no undefined weak symbol; control exactly `__capstone_hostcall` + `domain_main` | **refuted as first linked**: 2 undefined weak symbols (GLib's LeakSanitizer hooks). Holds after glib-0007. The control fired as corrected before the link |
| P1.2 | `code_len` 69–76 MiB, i.e. 29–36 MiB without the arena | **refuted**: 66.3 MiB, 26.3 MiB without it; the plan's estimate was high |
| P1.3 | a 128 MiB block, order 15, not doubled | confirmed (dmesg, every run) |
| P2.1 | M1 REACHED | confirmed from attempt 2; attempt 1 halted in `exit()` (fini walk), not predicted |
| P2.2 | M2 REACHED; heap peak_end 17–27 MiB | REACHED from attempt 3 (attempt 2 halted in `pthread_cond`, not predicted); heap **refuted narrowly**: 16.7 MiB |
| P2.3 | M3 REACHED | confirmed |
| P2.4 | M4 REACHED, frame 1 on stdout | confirmed, frame 1 byte-identical to stock's |
| P2.5 | M5 status 0; heap 26–32 MiB | confirmed: 26.8 MiB, under the 40 MiB arena |
| P2.6 | stderr: exactly the minimal build's 194 lines from M2 on; none in M1 | confirmed |
| P2.7 | output MATCH for every stage | confirmed |
| P2.8 | unserved: uname ×2, sysinfo, rt_sigaction, getrandom ≥1 by M2; rt_sigaction +4 by M4; nothing else | **refuted**: uname ×3 by M2 and ×5 by M3; no sysinfo; getcwd (17) and rt_sigprocmask (135) appear; rt_sigaction +3 by M4. Not pursued: output matched regardless. The unserved calls change nothing visible on this workload |
| P3.1–P3.3 | workload, flips and dns-ooo MATCH; flips fire | confirmed |
| P3.4 | ntp DIFFERS | confirmed, twice (one earlier ntp section stalled before its domain started; below) |

## Method

- **Images:** `host/cross-build.sh`, then `host/build-domain.sh`: M0's commit, with GLib's
  glib-0008 and the constructor/destructor support from the two commits before this one.
- **Runner:** `host/run-qemu.sh`, with the domain's two streams split by `host/domain-stdout.py`.
  - Guest: a private rootfs (`host/make-rootfs.sh`) with the a74a856 module.
  - `CAPSTONE_GP_NONLIN=1`; one boot per mode.
  - Host program: musl-capstone's `libc_test_host.c`.
- **References:** stdout against native stock tshark; stderr against the native minimal build,
  which has the same whitelist and a byte-identical generated `dissectors.c`. Both are made on the
  host from the bytes the guest was given.
- **Validation:** the runner was run on stand-ins before any tshark image (`predictions.txt`,
  section 0). Every verdict path was shown to fire both ways.
- **Every attempt is kept:** under `/tmp/capstone/tshark-app/runs/`, with each boot's image hashes.
  Stalls are listed below and never counted.
- **Reproducing the images.** `SHA256SUMS` holds the hashes of the images the counted boots ran.
  The runs directory keeps a copy in `counted-images/`, recovered by rebuilding with the counted
  sources; the rebuild matched all five hashes, so the build is deterministic.
  - After the boots, the committed `glib-0008` had a code comment corrected, and only the comment:
    the audit found its claim about the caller unevidenced.
  - Built from the committed sources, the images therefore differ in debug information. With
    symbols and debug information removed (`llvm-objcopy --strip-all`), all five are byte-identical
    to the counted images; M4 against M5 differs, so the comparison can fire.
  - `SHA256SUMS.stripped` checks a rebuild from the committed sources: strip each image to
    `tshark_mN.stripped.dom`, then run `sha256sum -c SHA256SUMS.stripped`.

## Stalls and discarded boots

Not counted, and every log is kept:
- **`-bbad`, `-yOUM`, `-kCjg`:** stages attempts 1–3, the table above. Attempt 3's images predate
  the hook's unserved report.
- **`-UkL7`:** the oracle as one ten-capture boot with `cma=2816M`. The guest spun at 100% CPU after
  login, while copying the image from the share, for 28 minutes. No domain had started; stopped.
- **`-nhXp` and `-1l9h`:** boot-login stalls ("OK", then silence), the known stall class.
- **`-Amjk`'s ntp section:** the guest stalled **before the domain started**, and it was stopped after
  26 minutes. The section logged nothing between its BEGIN marker and QEMU's termination.
  - Every section where a domain ran logs the monitor's region-share marker (`Print =
    Scalar(0x52434c4d)`) right after BEGIN; the audit counted the 62 other sections, halts included.
    That marker is printed while the host shares its regions, before it first calls the domain
    (`libc_test_host.c:49-50` against `:57`).
  - So the stall was in the guest's copy from the share, in starting the host program, or in
    `create_dom`. `-UkL7` fell silent at the same kind of step.
  - **CORRECTED:** at first I read this as the domain spinning for 26 minutes without hostcalls,
    because its 900 s alarm never fired. The alarm cannot tell the two apart: it is armed only once
    the host program runs, and a blocked guest step swallows it either way.
  - Two later ntp runs (`-CVwR`, `-lFvW`) returned in seconds with the predicted output.
  - Context, not a cause: another lane's memcached/memtier benchmark was running on the host.
- **Stand-ins:**
  - `-lfWY` stalled at boot-login;
  - `-2zpQ` booted the real M5 image by mistake (the image directory was not set), before the hook
    change; it returned status 0;
  - `-XXes` is the valid stand-in run.

## What this does not establish

- **The board.** Every domain on this ABI relies on QEMU fabricating a cursor-0 gp that cannot exist
  on silicon (CAPSTONE_GP_NONLIN=1 keeps it the type the entry glue gives it). A pass shows the
  program is correct under capability enforcement; the board needs the gp-captable ABI.
- **Safety.** level0 is one arena without per-object bounds and without revocation. This is a
  compatibility result.
- **Other captures and options:** only these captures, and only `-r -V -n`.
- **The runtime fixes outside this port.** C-64 and I-11 were fixed in the shared runtime on
  2026-09-25 (`docs/history/25-09-2026_01-30-00_c64-i11-runtime-fix.md`), and this port's own
  copy of the constructor support was removed. The stages and the oracle were rerun on that
  runtime, with the same verdicts. musl-capstone's `pthread_cond_t` (C-65) is still too small for
  its own fields. A patch is proposed and awaits the lead's decision; GLib avoids it through
  glib-0008.
