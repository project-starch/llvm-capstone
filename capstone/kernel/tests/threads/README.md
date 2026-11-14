# Trap threads and futex

Apply `series.s8-threads` after the step 3 trap patches and the step 5
user-memory and binder patches. The fragment contains 0050 and 0051.
It does not change the main series while those dependencies are in flight.

0050 admits protected same-mm threads with CLONE_VM, CLONE_SIGHAND,
CLONE_THREAD and CLONE_SETTLS. Each child receives its own tagged register
slots. Its sp and tp come from the clone argument references. The kernel
checks those references before publishing the child. Fork and clone3 stay
unsupported. The lifetime table belongs to the mm and is released by
`arch_exit_mmap`, after the final mm user exits.

The task keeps a separate 16-byte clear-child-tid reference. Clone and
set_tid_address copy into it with LDC/STC. `mm_release` stores zero through
that reference before clearing the task flag. This covers normal exit and
fatal signals. A failed store is dropped. Wake uses the stripped address.

0051 uses one aligned reference load for each futex sample, including the
fault-enabled retry. Wake-op atomics and compare-exchange use t6 as a reference.
The fault-in path performs atomic add-zero through the reference, so a stale
word returns EFAULT instead of retrying forever. Keys discard the handle
number. Protected futexes are private. The binder distinguishes timeout
references from requeue and wake-op integer counts. It refuses PI commands
and converts restart errors to EINTR so a handle cannot outlive its call.

## Build checks

On p13, use the stream's directory `~/asr-work/s8t`. Source
`capstone/tests/capstone-test-env.sh` before builds or tests.

```sh
bash capstone/kernel/tests/threads/run.sh /path/to/applied-linux
bash capstone/ports/musl-capstone/build-trap-smoke.sh "$CAPSTONE_TMP_ROOT/thread-smoke"
```

The host test compiles the actual binder from the applied kernel. It checks
clone pointer positions, the futex timeout/count distinction, refusal of
forged handles, PI refusal and set_tid_address. Its control gives requeue
the generic timeout-pointer mask. That control must fail.

The libc smoke creates two workers. Each updates its own object and TLS
counter across 1,000 yields. A mutex protects a shared counter. Both joins
must return their original object pointers, and the program must exit zero.
The build checks its link and the tag-preserving clone assembly. Linking
alone does not establish that it runs.

The modified kernel objects and the full Linux Image build on p13 with the
existing RISC-V Buildroot GCC toolchain. The Capstone LLVM installation used
for libc is `~/lanes/gatefix/llvm/bin`. Its source freshness cannot be checked
because that installation has no build.ninja. These are build results only.

## Integration contract

The build used base kernel patches 0001 to 0012, step 3 patches 0013 and 0014,
and a snapshot of the step 5 sources from its p13 Linux tree. The step 3
interface commit was `246c120b7b9`. Step 5 had no published implementation
commit at the time of the snapshot. Apply this fragment to those interfaces,
then resolve any later changes in the binder and Makefile together.

Required step 3 interfaces are `sublet_user_slot(task, n)`, the per-task
`capstone_slots[32][2]`, `mm->context.capstone_namespace`, and
`mm->context.capstone_root`. A child returns through the ordinary protected
ret_from_fork path. Its a0 slot contains the integer zero. Step 3 must preserve
t6 across S traps and fix up reference faults in `.sublet.uaccess.text`.
If step 3 adds `arch_release_task_struct`, combine its cleanup with 0050's
slot and exit-word clearing.

Required step 5 interfaces are `sublet_handle_slot`, `sublet_handle_address`,
`sublet_handle_guarded`, `sublet_get_user`, `sublet_uaccess_protected`, and the
binder's command-specific pointer mask. The general copy routine can move a
four-byte value with byte accesses. Futex therefore uses its own word load.
No change to the external handle format is required.

## Remaining plan v5 work

The implementation does not yet record a waiter's lifetime in `futex_q`.
Linux can count a stale waiter toward the requested wake count. Reference
loads still prevent a stale waiter from reading a replacement object, but
this does not satisfy the plan's live-waiter counting rule. Requeue races,
revocation during blocked IO, fatal-signal exit, invalid stack references
and stale wake-op need guest execution. Fatal Linux signals normally kill
the thread group, so a surviving-joiner test needs a kernel test hook.
Caught signals, robust-list translation, PI futexes and pthread cancellation
remain outside this slice. Raw runtime contexts have no join protocol.
Use pthread_create and pthread_join for libc-managed threads.

For the required stale-read control, retain a reference to a futex object,
end its lifetime, and place a fresh word at the same address. A kernel test
must call `futex_get_value_locked` through the stale handle and require EFAULT.
In a control build, strip the handle and perform the read by integer address.
With supervisor hardening disabled for that control, it reads the new word
and must fail the oracle. With hardening enabled, the raw read itself faults,
so it cannot demonstrate the bypass. This experiment has not run.

For the wake-op retry control, replace reference add-zero in
`fault_in_user_writeable` with address-only fault-in. A stale wake-op must
then hit a short timeout instead of returning EFAULT. Run this only in a
separate control build. The production build must return EFAULT promptly.
