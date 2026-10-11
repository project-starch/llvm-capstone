# tshark on the current delegated stack: stages and oracle, no unserved syscall (2026-09-30)

**Question.** Does the minimal tshark (Wireshark 4.6.8, `dissector-whitelist.txt`) still reach its
five stages and print stock's output on the delegated runtime as it stands after the signal,
row and socket lanes (`delegation-sockets`, over #135, #139, #140), and does the domain leave
any syscall unserved?

**Verdict.** Yes, and none. M1 to M5 REACHED and MATCH on dhcp; the oracle passes on dhcp,
dns_port, http, arp, their four flipped copies and dns-ooo, stdout byte-identical to native
stock tshark, and ntp DIFFERS as the negative control must. Every run's own report says
`unserved=none`, the launcher's counter `refused=0`. On 2026-09-24 the same images left
`uname`, `getrandom`, `rt_sigaction`, `getcwd` and `rt_sigprocmask` unserved; each is a row or a
runtime service since.

## What "restoring the dependency build" turned out to be

The default work directory `/tmp/capstone/tshark-app` is gone, which is what the
application-memory lane recorded as "12 unavailable tshark attempts". The full build of PR #128,
made with the qualified compiler on the ABI-v2 SDK, is intact under
`/tmp/capstone/delegation-ports/tshark-source`: the seven libraries (`deps-cap`), the cross build
(`xbuild`, `xsrc`), the native stock tshark (`native-stock/run/tshark`, 4.6.8 `e677bf052328`) and
the captures. Its compile commands embed that path, so it is used as `TS_WORK` rather than
linked into a new directory (a first attempt over symlinks failed `build-domain.sh`'s check that
`tshark.c.o`'s command ends in `$TS_WORK/xsrc/tshark.c`). Nothing was rebuilt but what the
runtime change requires: `deps/env.sh` keyed a new libc and runtime from this tree, and
`host/build-domain.sh` relinked the five stage images and the thirteen fixtures, 16 seconds in
all. #128's level0 images were moved aside as `domain.pr128-level0`, its sublet and chunks
images untouched.

`host/run.py` gained `TSAPP_DOMAIN_ENV`, extra `NAME=value` pairs for the domain: since #139 the
runtime prints its unserved report only under `CAPSTONE_DELEGATE_STATS=1`.

## Setup

- Compiler: `/tmp/capstone/delegation-ports/compiler-qualified`, clang 22 at `7d01722aab88`.
- Launcher: the sockets lane's `capstone-exec`, sha256 `546da6eda2952cb0f6995d907c0e32b41460fd34…`,
  built from the same runtime sources as this tree; monitor, QEMU and module as in
  `runtime/tests/application/results/20260930-sockets.json`; guest `--cma-mib 1024
  --process-cache-mib 768`, a private copy of the root filesystem.
- Images (level0 arm, 40 MiB arena, 1 MiB declared stack): `tshark_m1.dom` … `tshark_m5.dom`,
  `code_len` 70,313,440 bytes (67.1 MiB), block order 15 = 128 MiB; `tshark_m5.dom` sha256
  `974481ab38ce0797c90c4e4996abad2f6969226968077e3d062bfeb687b80ec9`.
- Runs: `runs/delegated-69juxuu0` (stages), `runs/delegated-xbis3qsz` (oracle), under the work
  directory.

## Stages on dhcp.pcap

| stage | exit | delegate rounds / syscalls | level0 peak_end | unserved |
|---|---|---|---|---|
| M1 `main` | 101 | 5 / 3 | 33,744 B | none |
| M2 `epan_init` done | 102 | 259 / 256 | 17,520,928 B | none |
| M3 capture opened | 103 | 329 / 326 | 25,975,152 B | none |
| M4 first frame | 104 | 336 / 330 | 28,117,568 B | none |
| M5 all frames | 0, stdout MATCH | 339 / 333 | 28,119,520 B | none |

On 2026-09-24: M2 `uname ×3, getrandom ×2, rt_sigaction, getcwd`; M3 `uname ×5, getrandom ×4,
rt_sigaction, getcwd`; M4 and M5 those plus `rt_sigaction ×3` and `rt_sigprocmask`; M5's
peak_end 28,123,936 B.

## Oracle

| capture | verdict | rounds / syscalls |
|---|---|---|
| dhcp, dhcp.flip | PASS, MATCH | 339 / 333 |
| dns_port, dns_port.flip | PASS, MATCH | 341 / 335 |
| http, http.flip | PASS, MATCH | 338 / 332 |
| arp, arp.flip | PASS, MATCH | 338 / 332 |
| dns-ooo | PASS, MATCH | |
| ntp | PASS, stdout DIFFERS (outside the whitelist) | |

Every flip changed stock's own output (the changed-input control). The whole pipeline, relink
to oracle, took three minutes of wall time.

## What this does not show

- The sublet and chunks arms were not relinked; `TSAPP_HEAP=sublet host/build-domain.sh` over
  the same work directory would.
- Nothing about the socket rows: the offline tshark opens no socket. What they would allow,
  name resolution without `-n` from `/etc/hosts` and live capture with `dumpcap` as a native
  child, is not attempted here.
- QEMU only, as the port's plan states.
