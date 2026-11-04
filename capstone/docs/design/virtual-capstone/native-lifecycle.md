# Native task lifecycle, signals and debugging (lane L, version 1)

Status: executable model plus negative controls, default off, not linked into
`capstone_vm.ko`. Header `runtime/include/capstone/native-life.h`, model
`runtime/virtual/tests/native-life/life.c`, controls `test.c`, benchmark
`lifebench.c`. It sits on the context double in `native-context.md` and
changes nothing there. Contract items: I5 (signals, ptrace), I6 (fork, exit),
I9 (frame layout, untouched).

The model fixes the ORDER of operations, the exclusion they need and the
state the kernel must keep for itself. It decodes no capabilities, has no
Linux scheduler and no real page tables. Where a production backend must
supply something the model cannot, the section says so.

## Objects

* Namespace: lifetime table (the srevroot: records of generation, state and
  free-list link, bump pointer, free list head), the address space (pages
  with per-granule tags, copy-on-write flags), the arena list, the hardware
  authority cache, thread references, an exclusion gate, an in-flight access
  counter. One per address space.
* Task: one cn context (a thread) plus kernel-held copies of what the thread
  registered with a capability: signal handlers, thread pointer,
  clear_child_tid word, robust list head. A kernel-held copy is taken from
  the thread's own register file at registration time, must be tagged,
  live and carry the needed right, and is revalidated at every use. A scalar
  address is never accepted where a capability is required (sigaction,
  set_tid_address, SETTLS, robust list).

## fork (I6)

0. A fork waits (`-EAGAIN`) while a checked-copy chunk is in flight: the chunk
   holds a raw page pointer, and sharing that page would let it write into the
   child's snapshot.
1. Siblings are held out: the namespace gate is raised before the first step
   and dropped after the last. A sibling's mint or retire sees `-EAGAIN`
   (a production sibling sleeps on the context lock; single hart does not make
   this unnecessary because the sibling may be preempted between the steps).
2. Address space: every page is shared and write-protected; the first write on
   either side copies the page together with its tag words (tag-aware COW).
   Anything that writes through a shared page (ptrace poke, a checked copy)
   breaks the sharing first.
3. Namespace: a NEW table is allocated and the used prefix of the records is
   copied, so every slot, generation and free-list link is identical. The
   child therefore holds exactly the capabilities the parent held and nothing
   aliases: a revoke in the child writes the child's table only.
4. The copy is not atomic with respect to a sibling unless the gate is held.
   A torn copy (records from before a retire, header from after) gives a free
   list that names a live record, and the child's next mint reuses a
   (slot, generation) that an existing capability still names.
5. The child context is created last. A failure after the namespace was cloned
   rolls the clone back: every page reference is dropped, the child table's
   cached authority is invalidated and the table released (its root can be
   handed to the next namespace, so a stale cache entry would be honoured).

Threads (clone with CLONE_VM) do not clone anything: they take a reference on
the namespace and get their own context. The child's stack and thread pointer
come from capability slots of the parent and are checked (tag, lifetime,
rights); a scalar thread pointer is `-EINVAL`.

## exit (I6)

Per thread: robust list, clear_child_tid and its wakeup, futex waiters,
context destruction, namespace reference. Per process, when the last thread
is gone, in this order:

1. refuse new accesses (`dying`), drain the in-flight checked-copy chunks
   (the caller sleeps and retries on `-EAGAIN`),
2. invalidate the cached authority of this table (CSFENCE.REV) while the
   table still exists,
3. free the table once, whatever the number of arenas,
4. release the address space (page puts; tag clearing belongs to the page
   free path and is gated on the may-hold-tags flag).

There is no per-arena CSRETIRE loop and the namespace teardown clears no
tags. Skipping step 2 leaves entries that name a table root which the
allocator hands to the next namespace.

## signals (I5)

* A handler is registered from a code capability in a register slot.
* The registered handler is revalidated (tag, lifetime, execute right) at
  every delivery; a dead handler is never installed as PCC.
* Delivery pushes the full interrupted frame (all capability registers with
  tags, FP, PCC, PC, pending service event) onto the kernel-held stack of the
  context double (`cn_signal_push`, eight levels) and builds the handler frame
  from the kernel's own copy. What the thread sees in memory is a sigcontext
  with the payload words of the registers, an informational tag hint and one
  token (nonce, depth, context id). The tag hint and the PC field are not
  inputs.
* `rt_sigreturn` authenticates the token against the kernel-held top frame
  (context, depth, single-use nonce) and restores the kernel's frame. Only
  integer registers that were plain scalars when the signal arrived accept the
  handler's edits (Linux lets a handler change them, so does this). Capability
  registers, PCC and PC are not editable in v1: a handler that wants to resume
  elsewhere is not supported (UNRESOLVED, needs a PCC-bounded set-PC).
* Restart: `nl_sig_restart_policy` is the Linux riscv table and
  `nl_sig_finish_service` applies it. A service that was pending when the
  signal arrived stays pending across the handler (the context double saves and
  restores the event). EINTR is a reply to that same event. A reissue hands the
  dispatcher the six argument registers of the PROTECTED frame: capability
  arguments are the kernel's records (a handler cannot retarget a buffer),
  scalar arguments carry the handler's edits as in Linux.

## ptrace and /proc/PID/mem (I5)

* A confined native task is never a debugger and never opens another task's
  `/proc/PID/mem`. Policy is a function of the actor, not of the address.
* An ordinary debugger with the right uid on a stopped tracee reads data
  words. A capability's tag never travels in a read; a poke is a scalar store
  that clears the granule's tag, breaks copy-on-write first, and a register
  write goes through the scalar path that clears the register's tag.
* The debugger's register view shows payload words and tag hints only. It does
  not show signal tokens.

## TLS, futex, robust list

* Futex keys are (namespace, address). Shared futexes are not supported
  (fail closed). Wait reads the word through the thread's capability and
  needs read right; wake needs a live capability.
* The robust list is walked at exit with the authority the program stored:
  every entry pointer is loaded WITH its tag and used under its own bounds; an
  untagged, dead or too-short entry ends the walk; the walk is bounded.
* clear_child_tid is written through the registered capability only if it is
  still live and writable. A retired mapping is never written.

## What the controls prove and what they do not

Each control passes on the model and fails on a named broken variant
(`run.sh` builds and runs all of them; `NL_MUTATE_*` and the context double's
`CN_MUTATE_*`). They prove the ordering and policy rules above on a software
model. They do not prove hardware behaviour of CSRETIRE, the cache, tag COW in
the kernel, the arch signal path or ptrace on the real kernel.
