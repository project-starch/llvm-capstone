# Native-task kernel series for Linux 6.4.14 (the boards' kernel)

The boards run the buildroot kernel, Linux 6.4.14 (upstream, unpatched, `build-fpga/.config`). The native adapter's
kernel side was written for Linux 6.1. This directory is that series ported to 6.4.14. `install-native-task.sh` applies
it, in the order of `series`, to a pristine 6.4.14 tree: the tag-aware memory patch, mprotect, gcc, then the seven
patches `../install-native-task.sh` applies on 6.1. It stages the built-in adapter the same way. It no longer turns SMP
off (see "Configuration"). The patches apply with zero fuzz. The script refuses fuzz and any
other kernel version.

## Configuration: the minimal delta

On the board's stock `.config`, olddefconfig after the install changes exactly these lines (`native-config.delta` in
the board kit): `RISCV_CAPSTONE_TASK=y`, `VIRT_CPU_ACCOUNTING_GEN=y` in place of `TICK_CPU_ACCOUNTING`, and
`CONTEXT_TRACKING_USER(_FORCE)=y`. SMP, NR_CPUS, TREE_RCU, HZ and preemption stay as stock.

- **One possible CPU, not `!SMP`.** The 6.1 Kconfig required `!SMP`. The requirement in the code is one POSSIBLE CPU:
  - `capstone_vm.c` refuses to load native transport, checked copy, binfmt or unpinned backing when
    `num_possible_cpus() != 1` ("Tag-copy interrupt exclusion and namespace sweeps are currently single-hart protocols.
    Offline-but-possible CPUs are not safe");
  - `captag.c` refuses the same way;
  - `checked-copy-dom.h` requires it on every copy.

  The 6.4 Kconfig drops `!SMP`, and `riscv_capstone_attach` now refuses on more than one possible CPU too. The board's
  device tree has one hart (`cpu@0`), so an SMP kernel has one possible CPU there. `nr_cpus=1` gives the same on a
  machine with more harts. Two harts cannot be tried on QEMU, which refuses its own way: "x-capstone-u-mode requires
  exactly one possible hart".
- **Generic virtual CPU accounting and forced user context tracking stay.** They are required: on an SMP kernel with
  stock tick accounting, the gate fails exactly two checks.
  - task-driver's `CHECK("compute charged to user", before > 0)` fails, reading `compute usage user=0.000000
    system=0.501327`.
  - Its `CHECK("compute resumes", user_ticks(pid) > stopped)` fails too.
  - Both read the task's user ticks (`tests/native-task/task-driver.c:70-76`).
  - This matches the design's "Linux must account domain computation as user execution"
    (`docs/design/virtual-capstone/native-linux-tasks.md`).
  - The Kconfig keeps `depends on VIRT_CPU_ACCOUNTING_GEN && CONTEXT_TRACKING_USER_FORCE`.
- `NATIVE_TASK_6_1_CONFIG=1` also turns SMP off, as the 6.1 recipe did.

## How the port was made

Each 6.1 patch was applied to a pristine 6.4.14 tree in turn, and each rejected hunk was rewritten by hand. The 6.1
pre-images came from reverse-applying the series to a 6.1 tree. For the 33 touched files they are byte-identical to the
QEMU twin kernel's 6.1.0 sources. Every hunk that applied with an offset was checked against its enclosing function.
The 6.1 and 6.4.14 versions of these functions are identical: `search_binary_handler`, `exec_binprm`, `kernel_execve`,
`kernel_clone`, `do_wait`, `child_wait_callback`, `do_select`, `do_fcntl`, `check_fcntl_cmd`, `f_dupfd`, `alloc_fd`,
`create_pipe_files`, `do_sigaction`, `cp_new_stat` and `chmod_common`. `fcntl_getlk` and `fcntl_setlk` differ only in
`locks_inode` becoming `file_inode`. The 6.4 versions of `do_exit` and `bprm_execve` gain lines that no hunk touches.

## Hunks that are not mechanical

| 6.1 patch | Hunk | Why it does not apply | 6.4.14 equivalent |
|---|---|---|---|
| unpinned | `asm/page.h` clear_page | 6.4 adds a Zicboz `clear_page` (`lib/clear_page.S`); the board config has `RISCV_ISA_ZICBOZ=y` | `clear_page` is `capstone_clear_page` in both configurations. Its fallback is 6.1's `memset`, so cbo.zero runs only under `captag_bulk_on`, as on 6.1. The unused Zicboz `clear_page` is still built. |
| unpinned | `mm/Makefile` | context: 6.4 adds `pgtable.o`, `pmem.o` | the same `obj-y += captag.o` line |
| task | `kernel/Makefile` (also spawn's) | context: 6.4 adds `pi/` | the same object line |
| task | `thread_info.h` (fuzz 1) | 6.4 dropped `TIF_SYSCALL_TRACE/AUDIT/SECCOMP` | unchanged; `TIF_CAPSTONE` = 12 is still free |
| task, spawn | `_TIF_SYSCALL_WORK` in `capstone_task.c`, `capstone_spawn.c` | riscv 6.4 uses generic entry, and that work moved to `thread_info->syscall_work` | `riscv_capstone_syscall_work()` (`asm/capstone_task.h`) refuses when ANY syscall_work bit is set. That is the 6.1 set (trace, tracepoint, audit, seccomp) plus 6.4-only syscall user dispatch, SYSEMU and exit trap. |
| task | `entry.S` trap hook | 6.4's `handle_exception` dispatches to C handlers, and they call `irqentry_enter` | `call riscv_capstone_trap_enter` just before the dispatch, after `gp` is loaded. That is the point 6.1 used (before tracing), so it still runs before any C handler. |
| task | `entry.S` return hook | 6.4 has no asm work loop: `exit_to_user_mode_loop` runs it in C | `asm/entry-common.h` sets `ARCH_EXIT_TO_USER_MODE_WORK` to `_TIF_CAPSTONE`, so the loop keeps running while the flag is set. `arch_exit_to_user_mode_work` steps the task only in an iteration whose IRQs-off flag snapshot held no other exit work, which was 6.1's priority. It calls `riscv_capstone_return` with IRQs off, as 6.1's entry.S did. |
| files | `fs/open.c` ftruncate | 6.4 uses `security_file_truncate(file)` and `file_mnt_idmap(file)` | the held-file variant uses the same two calls as 6.4's own `do_sys_ftruncate` |
| directory | `fs/namei.c` link | 6.3 replaced `mnt_user_ns` with `mnt_idmap` | `mnt_idmap`, as 6.4's `do_linkat` uses it |
| helpers | `fs/open.c` hunk 2 | fails only against the 6.1-shaped ftruncate | applies to the ported one |

The built-in adapter (`../module`) has 13 call sites of APIs that 6.3 changed. They now go through `cv_mnt_idmap`,
`cv_owner_is_fsuid` and `cv_syscall_work` in `capstone_vm.c`. On 6.1 each macro expands to the expression it
replaced. The 6.4 board kernel, the 6.4 QEMU kernel and a 6.1 control kernel built from the same adapter sources all
compile with no warning outside the adapter's pre-existing `-Wdeclaration-after-statement` warnings.

## QEMU gate (8-9 Nov)

All runs used `run-musl-io.py io` with the same inputs: QEMU d2bac4e08a (`8f3f9118`), QEMU's `fw_dynamic` (`2335a286`),
rootfs `9903242c`, the same probes and one hart.

| Kernel | Image | Result |
|---|---|---|
| 6.4.14, this series, the 6.1 native config moved forward (SMP off, TINY_RCU, generic accounting) | `9c0e6eff0248432f` | PASS: MUSL_IO 170/0, 4/0, 176/0, 4/0; native integration 51/0; gcc coverage 20/0; entry and task failures 0 |
| 6.1 + the 6.1 series, the same adapter source (control) | `63642abef1915cbb` | PASS, the same readings; native_route counts per syscall number identical (840) |
| 6.4.14, SMP=y, TREE_RCU, stock tick accounting, no context tracking | `313c7e405a79056f` | FAIL: "compute charged to user" and "compute resumes" (`compute usage user=0.000000 system=0.501327`); every other check passes |
| 6.4.14, SMP=y, TREE_RCU, generic accounting + forced user context tracking (the minimal delta) | `31296c3bdb9cdbcb` | PASS, the same readings; `compute usage user=0.497410 system=0.003977` |

## Not covered by the port

- **Generic entry runs the loop on every exit to user mode of a native task.** The QEMU gate exercises this path, but
  only on one CPU model.
- **6.4's `do_exit` adds `WARN_ON(irqs_disabled())`.** The QEMU gate rejects any `WARNING:`, and none appeared.
- **Page copies.** captag hooks the same four entry points as on 6.1. New 6.4 page-copy paths that bypass
  `copy_highpage`/`copy_user_highpage` were not enumerated. THP and KSM are off in both configs.
- **The accounting options cost something on every user entry and exit.** That cost applies to every process on the
  native kernel, Plain included. A Plain run on the stock kernel therefore differs from one on this kernel by those
  options. Compare native with Plain on THIS kernel.
