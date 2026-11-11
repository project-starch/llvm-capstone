# zend-alloc-repros on the virtual profile, 2026-11-11

The first run of this corpus's two virtual arms, and a run of the nested allocator layer after
it moved to the one-operand CREVOKE (`heap_end(ptr)` in `ports/php/zend-alloc/sublet/zend-cap-mode.c`).
One invocation of `php/tools/run-php-corpus.py --arm virtual-malloc --arm virtual-nested-pools`,
one boot per arm, every case its own `capstone-vexec` process of one image at the arm's
`PHP_CAP_MODE`. No earlier virtual record of this corpus exists to compare with; all 30
defects are also cases of `../../../engine-repros` (by fix commit), whose record of the same
date reads them on the same image.

| arm | configuration | controls | CAUGHT | MISSED | NO-READING |
|---|---|---|--:|--:|---|
| `virtual-malloc` | `php-real-stock` (mode 0) | `revocation-sanity` completes | 0 | 0 | not-reached 20, out-of-denominator 10 |
| `virtual-nested-pools` | `php-real-exact` (mode 7) | `revocation-sanity` completes, `nested-bound-probe` faults | **20** | 0 | out-of-denominator 10 |

`sensitivity/` holds the same boots scored with the native crashers in: `virtual-nested-pools`
28 CAUGHT and 2 setup-fault, `virtual-malloc` 1 CAUGHT, 23 not-reached and 6 setup-fault.

**The nested arm.** All 30 runs end in a cause-25 fault (a load or store through an ended
lifetime). 20 land in a function the case declares and are CAUGHT. The other 10 are the cases
whose host record is a signal on the unprotected build, out of the denominator by the default
scoring whatever the arm did; 8 of those 10 faults are at a declared site too, and two are not
(case 15 in the libc's write path, `dl_call_once`; case 18 in `zend_do_fcall_common_helper`).

**The baseline arm reads no MISSED, and that is the runner, not the arm.** The 20 cases the
nested arm catches ran to their normal end at mode 0 with `revocation-sanity` complete and a
host record that completes. The runner establishes reach from "the baseline arm ran this case
to its normal end", read from the completed set of an *earlier* arm of the same invocation;
while the baseline arm itself is observed that set is still empty, so each of its silent runs
is judged `not-reached` (`silent, but reach is not established: sanity=ok, baseline
completed=False`). Re-observed offline from the stored transcript with the arm's own completed
run counted as the baseline run, the arm reads 20 MISSED and 10 out-of-denominator. The bundle
here is what the runner wrote and is not edited. The remaining 10 are the native crashers: 7
fault at mode 0 with cause 24 or 28 (the program's own corrupted value or overrun, as on the
host), one aborts (134), one ends with its designed exit 255, one completes.

The baseline boot ran twice (a first invocation was stopped before its second arm for a
queueing reason and the invocation was repeated whole): the two baseline boots agree in every
verdict, reason and cause.

## Platform, by content

| component | sha256 |
|---|---|
| image `php-real-nested.dom` | `a8a77eadd554cc8577e139654d634270d21052294bd0d75e9cecf05789a87949` |
| `probe-nested.dom` (the bound control) | `499c38174334c5f0fb63d9ead196f9e3032937638a11d1d32a227a39d14a2d07` |
| `qemu-system-riscv64` (one-operand CREVOKE, CRENEW) | `2d5a7107890e48475fada73439d90ffd5c7cff166eb1d547d7a9fc5f31722365` |
| `capstone-vexec` | `0f8e0f903725d734817b3e30d79127c290f3480a641d4f6de8bc7f699d05c76e` |
| `capstone_vm.ko` | `e47bca62addfc9356f8150755f2500b737912eb01088e91398e29a1ed7a4e180` |
| `Image` / `fw_jump.elf` / `rootfs.ext2` | `48eb725a...` / `5b5c047a...` / `62049800...` |
| compiler (`clang`) | `0bbb1f88bc9084e387ec6b3c2db24876dfe267897a13f0d1a53d606a2ded189e` |

`-cpu` and module arguments are `ports/php/app/run-virtual.py`'s (`x-capstone-exact-bounds`,
`x-capstone-test-csrs`, 32 supervisor slots, `x-capstone-addr=sv39`; `node_initial_pages=1024
supervisor_slots=32`), a pinned load. The SDK has an 8 MiB stack, 16 MiB of data, line tables
in the musl objects and the corpus codegen gate. The image holds 7 CDERIVE and 4 CREVOKE
instructions, every CREVOKE of the one-operand form. The bundles' own `platform` block names
paths relative to the run's work directory, as the runner records them.
