# plain-temporal-repros on the virtual profile, 2026-11-11

The first run of this corpus's two virtual arms, on the nested allocator layer after it moved
to the one-operand CREVOKE (`heap_end(ptr)` in `ports/php/zend-alloc/sublet/zend-cap-mode.c`).
One invocation of `php/tools/run-php-corpus.py --arm virtual-malloc --arm virtual-nested-pools`,
one boot per arm, every case its own `capstone-vexec` process of one image at the arm's
`PHP_CAP_MODE`. No earlier virtual record of this corpus exists to compare with; 22 of the 26
defects are also cases of `../../../engine-repros` (by fix commit), whose record of the same
date reads them on the same image. Cases 18 to 21 are the four that are not.

| arm | configuration | controls | CAUGHT | MISSED | NO-READING |
|---|---|---|--:|--:|---|
| `virtual-malloc` | `php-real-stock` (mode 0) | `revocation-sanity` completes | 0 | 0 | infra 3, not-reached 11, out-of-denominator 11, setup-fault 1 |
| `virtual-nested-pools` | `php-real-exact` (mode 7) | `revocation-sanity` completes, `nested-bound-probe` faults | **8** | 0 | infra 2, not-reached 4, out-of-denominator 11, setup-fault 1 |

`sensitivity/` holds the same boots scored with the native crashers in: `virtual-nested-pools`
17 CAUGHT, 6 not-reached, 2 infra, 1 setup-fault; `virtual-malloc` 1 CAUGHT, 20 not-reached,
3 infra, 2 setup-fault.

**The nested arm.** 8 CAUGHT, each a cause-25 fault in a function the case declares. 11 are the
native crashers, out of the denominator by the default scoring; 9 of them fault with cause 25
and two abort (exit 134). Cases 10 and 26 are killed by the 40 s watchdog, as their
engine-repros counterparts CRASH-075 and CRASH-161 are on every arm. Case 7 faults with cause
25 in `strlen`, which it does not declare, on this arm and on the baseline alike (at mode 0
this layer ends no lifetime, so the one that ended is the libc heap's): a setup-fault, no
reading. **Cases 18 to 21 cannot run on this image**: they call `ext/mbstring` and `ext/pcre`,
the image is PHP configured `--disable-all`, and each ends in PHP's fatal error for an
undefined function (exit 255) on both arms, `not-reached`.

**The baseline arm reads no MISSED, and for the completed runs that is the runner, not the arm.**
As in `../../../zend-alloc-repros/results/20261111/README.md`: while the baseline arm itself is
observed, the set of cases "the baseline arm ran to its normal end" is still empty, so a
silent completed run is judged `not-reached`. Re-observed offline from the stored transcript
with the arm's own completed run counted as the baseline run, the arm reads 5 MISSED, 6
not-reached (cases 18 to 21 and two more that this route does not license), and the other
cells as written. The bundle here is what the runner wrote and is not edited. The third
baseline `infra` is case 12 (CRASH-081's trigger), which the watchdog kills at mode 0 only, as
in engine-repros.

## Platform, by content

The image, probe, QEMU, launcher, module, guest images, compiler, `-cpu` string, module
arguments and SDK are those of `../../../zend-alloc-repros/results/20261111/README.md`
(image `a8a77eadd554...`, QEMU `2d5a7107...`). This invocation ran beside two other boots of
the same work; its watchdog readings are the ones engine-repros has for the same triggers.
