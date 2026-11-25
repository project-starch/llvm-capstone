# Trap-route mapping reuse: QEMU qualification results (2026-11-25)

The rows of PREREG.md (pre-registered at afeace3b2e0d; the 963 MiB amendment f245755dcbe5, both pushed before the
boot) were run in one boot at 12:38. The platform is the one PREREG.md names. The probe images are the build of
that commit's sources against the eight SDK variants; their hashes are the boot's `probes.sha`. The control was
trap/eval's own smoke.dom, which printed `ok` and exited 7, as its source returns. Faults are the kernel's
`sublet: fault` lines.

The first attempt (12:30) produced no reading and is void. It was the harness, not the subject: the initramfs had been
repacked as a normal user, so buildroot's setuid busybox became setuid to that user and /init ran with euid 1000;
busybox there also has no `timeout`; and the loader refuses a cwd of "/".

## Kernel facts (renew-probe, variant `on`)

| row | reading | against the prediction |
|---|---|---|
| unmap-renewed | `RENEWED tag=1 cursor_same=1 byte=0x5a`, `RENEW-UNMAP rc=0` | as predicted, except `mapped=-1` |
| unmap-stale | `RENEW-STALE-UNMAP rc=-1 errno=14`, then the renewed munmap rc=0 | as predicted: the stale reference is refused (EFAULT) |
| read-stale | fault cause=25 at the old block's address, after `STALE-READ-BEGIN` | as predicted |
| many 40000 | `renewals=40000 of 40000 tag=1 cursor_same=1`, munmap rc=0 | as predicted, except `mapped=-1` |
| many 200000 | `renewals=200000 of 200000 tag=1 cursor_same=1`, munmap rc=0 | as predicted, except `mapped=-1` |

- **Not verified, the /proc check:** `mapped=-1` means `/proc/self/maps` could not be opened from the domain. The
  likely reason is the loader's cwd grant (resource.h takes the cwd as the domain's root); that is not traced. So "the range leaves the address space" was NOT checked; only
  munmap's return code was. The probe's exit status (1) reflects that, not a failed munmap.
- **Not verified, the out-of-place path:** whether either renewal count crossed the record's generation limit, and so
  went out of place, is not shown. Every renewal returned a tagged reference at the same cursor, and munmap by the last
  one succeeded.

## Safety (mapping-reuse-probe)

| row | variant | reading | against the prediction |
|---|---|---|---|
| control | off-nolife | hits 0, puts 0, `UAF-END`, exit 0 | — |
| stale200, stale16 | off-nolife | fault cause=25 | as predicted |
| control | on-nolife | hits 147, puts 147, same200=1, same16=1, exit 0 | the vexec record's 147 and 147 |
| stale200, stale16 | on-nolife | hits 147, puts 147, same200=1; fault cause=25 | as predicted: only the put's CRENEW can end the lifetime here |
| stale200 | norevoke-nolife | `UAF-STALE-READ-RETURNED stale200 0x33`, exit 3 | as predicted: the positive control |
| reuse | on | `UAF-REUSE-OK same_address=1 hit=1` | as predicted |
| reuse | off | `UAF-REUSE-OK same_address=1 hit=0` | as predicted |
| reuse | nozero | `UAF-FAIL reuse data of the previous block is visible` | as predicted |
| alias | on | hits 1, puts 1, fault cause=25 after `UAF-ALIAS-READ-BEGIN same_address=0` | the fault as predicted; `same_address` read 0 where 1 was predicted |
| protect | on, keepprot | `UAF-FAIL protect mprotect` | the pre-registered residual branch |
| saturate | on | hits 320, puts 480, untracked 0, `UAF-SATURATE-OK` | as predicted |
| saturate | live64 | hits 128, puts 192, untracked 288, `UAF-SATURATE-OK` | as predicted |
| realloc | on | hits 1, puts 3, evicted 1, `UAF-REALLOC-OK` | as predicted |
| pressure | on | 253 blocks of 16 MiB, stopped on enomem, puts 60, evicted 60 | as predicted: the cache was given back |
| pressure | off | 253 blocks, enomem | equal to on |

`renew_failed` read 0 in every row.

## Residuals

- **The mprotect condition is UNTESTED.** The trap kernel refuses mprotect through a heap object's reference, so the
  probe never reprotected a page. `__heap_protected` is linked and called from mapping.c, but no run has shown that a
  reprotected mapping is not reissued. On this route a program can only change a heap mapping's protection with the
  owner reference, which mallocng never hands out.
- **Alias: the fault is shown, the attribution is not.** The alias read faulted with cause 25 as predicted, but
  `same_address=0` means the replacement object did not land at the freed one's address. So the fault is consistent
  with the put's renewal (hits 1, puts 1), but this row does not show the reissued mapping under the alias. The
  stale200/stale16 rows do show it (same200=1 with the fault), and the norevoke control shows the read would
  otherwise return.
- **Out-of-place renewal and the address-space check:** see the kernel facts above.
