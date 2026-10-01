# Linux capability execution boundary (M1)

Status: **PROPOSED TARGET; QEMU ACCESS SLICE ONLY**, 2026-10-01. This chooses
the direction for the M1 prototype; it does not claim that Linux user-mode
capability execution or the required kernel and hardware contracts exist yet. It refines the
[trusted-Linux memory design](trusted-linux-memory.md) and the
[application compatibility M1 gate](../plans/trusted-linux-application-compatibility.md).

## Decision

Run capability-enabled applications as ordinary Linux **U-mode processes**.
Linux owns their virtual mappings, faults, scheduling and syscalls. Capstone
checks object authority on every application access, and the ordinary MMU
checks translation and page permissions on that same access. One process
address space has one Linux page-table root and one selected lifetime
namespace, shared by its threads. A pointer chooses an object, not a page-table
root. The current C-mode/delegated runtime remains a compatibility bridge
while this target is built; its exchange buffers are not the final syscall ABI.

The alternative is to extend C-mode until it reproduces Linux VM, page faults,
thread scheduling, signals and process cloning behind the monitor. That keeps
two address-space implementations and makes a completed buffer syscall depend
on a separate delegation service. It is useful for migration and experiments,
but conflicts with the goal that Linux implement ordinary OS behavior. This
decision is about the *target execution boundary*, not removal of the current
module or monitor from existing workloads.

The current QEMU cannot implement the target by changing libc alone. At the
repository's pinned `capstone-qemu` commit `ac2837aa0`, `PRV_C` equals `PRV_M`
(`target/riscv/cpu_bits.h:617-622`). The walker returns the input address with
full page permissions for M-mode **or** a capability access
(`target/riscv/cpu_helper.c:856-867`); capability access types are identified
in `include/hw/core/cpu.h:83-99`. These are source facts about that pin, not
measurements of the proposed path. U-mode enablement, access classification,
translation, fault routing and a corresponding RTL design are M1 work.

## State and transitions

Let `c` identify an address-space *instance*, not a PID or reusable ASID.
`P_c` is Linux's page-table root and VM areas; `N_c` is the protected object
lifetime graph. `K_c` is a non-user-writable selector for `N_c`. `G_t` is one
thread's tagged capability register state. The kernel saves and restores
`G_t`, `P_c` and `K_c` as one scheduling contract; a TLB ASID is only a cache
key and cannot substitute for `c` or for node generations.

| Event | Actor | Required transition |
|---|---|---|
| `switch(t1 -> t2)` | Linux plus hardware | Save `G_t1` with tags; install `P_c2`, `K_c2` and `G_t2`; invalidate or tag translation and lifetime caches so an authorization from `c1` cannot serve `c2`. Do not expose an intermediate user execution state. |
| `load(p,w)` / `store(p,w)` | U-mode hardware | Decode/tag-check `p`; check bounds, rights and liveness in `N_c`; translate its cursor through `P_c`; check PTE permissions; complete or raise a precise fault without a partial write. This applies to integer, FP, atomic and capability memory operations. |
| `page_fault(p)` | Hardware to Linux | Report virtual address, access kind and restartable instruction state. Linux may install/change a PTE and retry only if capability authority still holds. A page fault must not turn a dead or out-of-bounds pointer into a valid one. |
| `mmap(len,prot) -> arena` | Linux and capability ABI | Reserve a virtual range under `P_c`; return tagged authority for precisely that range to libc through a defined kernel/user ABI. A bare integer syscall return cannot manufacture a tag. |
| `malloc(n) -> p` | libc/allocator | Derive bounded object authority and a fresh lifetime in `N_c` from an arena. No syscall is needed when the arena has space. |
| `free(p)` | libc/allocator plus completion mechanism | Retire the object and descendants in `N_c`; wait for or cancel older checked accesses before reuse of its bytes or lifetime identity. No PTE change is required for an inner object. |
| `munmap(range)` / replacement | Linux plus lifetime interface | Retire authority affected by the range and complete older accesses before its virtual address or backing can name an unrelated object. Define partial-range treatment before implementation. |
| `read(fd,p,n) -> k / error` | libc, Linux, capability copy path | Validate the requested writable span; Linux performs the file operation and writes through checked authority. An invalid requested span returns `EFAULT` before consuming input under the chosen boundary contract. A blocked call must not copy into reused storage after concurrent `free`. |

The central lifetime contract is visible even when the allocator immediately
reuses the exact address:

```c
p = malloc(64);
free(p);
q = malloc(64);   // assume the same virtual address as p
*p;              // capability fault: retired object identity
*q;              // succeeds: fresh object identity
```

These are requirements on the two memory accesses, not on optimizing C code
with undefined behavior. `free` completes retirement before returning storage
for reuse. A new allocation must have an identity distinguishable from every
retained pointer to an older allocation, even at identical bounds and address.
The model advances a generation without wrapping; exhaustion refuses allocation.
`munmap` followed by `mmap` must preserve this distinction too. `malloc` within
an arena leaves Linux's VMA and PTE permissions unchanged; it cannot restore
write permission removed by Linux.

A private process clone needs a fresh address-space instance `c`, even when
the numerical PID or ASID is reused. It copies live and dead lifetime state
into the child's independently owned namespace: inherited `p` remains stale,
inherited `q` remains valid, and private writes stay private. An old context
with no live objects or VMAs is still an old instance. Reusing its storage
requires a separate instance-identity protocol; the bounded model instead
claims a previously unused context and never recycles it.

For `load(p,w)`, success therefore means

```text
tag(p) ∧ live(N_c, node(p)) ∧ rights(p, read)
∧ [cursor(p), cursor(p)+w) ⊆ bounds(p)
∧ translate(P_c, cursor(p), w) permits read.
```

Each interval calculation must reject overflow. These are logical predicates;
the hardware may overlap their implementation. If a capability spans pages,
every touched page needs the appropriate PTE permission. A linear capability
load that clears its source also requires a writable page and must be
restartable across a copy-on-write fault without moving authority first.

## Syscall authority and concurrent retirement

The first ABI slice should carry a tagged buffer capability to a trusted
kernel-side copy operation. Linux may use bounce storage internally, but the
copy must use the original caller authority or a kernel-issued reservation
derived from it; turning the cursor into an unchecked integer would lose the
object bound. For `read(fd, malloc(16), 4096)`, reject the 4096-byte request
with `EFAULT`, even if Linux would have returned fewer bytes. A nested argument
such as `iovec` requires separate checks for the descriptor array and each
element. This is the deliberately chosen Capstone boundary contract; it is
not asserted to be stock Linux's behavior for every syscall.

Preflight does not freeze an object lifetime. If another thread frees `p`
while `read` blocks, the final copy must either fail recoverably before
writing, or hold a lifetime reservation that prevents reuse until completion.
The first prototype can use a synchronous, copy-based syscall path. Before
M1 is closed, specify which actor owns a reservation, how cancellation
releases it, and how Linux converts a capability fault in a user copy to
`EFAULT` without killing the process. Direct I/O and asynchronous requests
need their own completion rules; a pinned physical page alone is insufficient.

## What must be demonstrated before M1 is closed

1. A bounded state model covers two address spaces with equal virtual
   addresses and node numbers, a context switch/ASID reuse, a pending checked
   store crossing `free` or `munmap`, and an `mmap`/page-fault retry. Include
   both harts with pending accesses to the same retiring context. Follow reuse
   with accesses through retained old and fresh pointers: the old must fault,
   the fresh must complete. Check that no stale or foreign authorization reaches
   a byte and that failed transitions preserve state. Counterexample variants
   should omit the lifetime-context change, generation advance, cache
   invalidation, and access drain separately; denying all new accesses must
   also fail the gate.
2. A QEMU prototype runs a small U-mode process with an ordinary Linux
   mapping, a fresh `malloc` object, a restartable page fault, preemption and
   `read(fd,p,n)`. Both object authority and Linux PTE permissions must be
   independently observable: disable one check at a time as a positive
   control. Exercise integer, FP, atomic and capability access paths, including
   tag-preserving register save/restore.
3. The prototype distinguishes `EFAULT` from a fatal domain fault for a
   checked kernel copy. It demonstrates `malloc(16); read(fd,p,4096)` without
   consuming the input and a delayed copy against retirement on two harts.
   If two-hart completion is deferred to M3, M1 may establish the single-hart
   path but must remain open on concurrent lifetime safety.
4. Record the actual QEMU/RTL, Linux, compiler, libc and ABI changes and their
   costs. Compare the integrated path with the delegated bridge on matched
   behavior. A source-level architectural choice is not evidence of a working
   kernel or cheaper hardware.

The first implementation experiment covers the access/fault path in item 2,
but not its Linux-process or context-switch gates. Adding more delegated
syscall shapes improves the bridge but does not settle M1.

The reviewed [executable boundary model](../../models/trusted-linux-m1/README.md)
covers the finite cases in item 1 with explicit per-hart invalidation and
drain transitions. Its [record](../../models/trusted-linux-m1/result.json)
contains 19 families, 90,735 named-event schedules and counterexamples for
18 distinct injected faults. It now checks the retained `p`/fresh `q` contract
after allocation and remap, private-clone namespace freshness, unchanged PTE
rights on object reuse, and pending accesses on both harts. Its globally
visible retirement break is an assumed contract, not an implemented mechanism.
Capability encoding, tag save, multi-page translation, kernel copy recovery
and the RTL implementation remain untested. Item 2 still requires a real
Linux process, including the same old/fresh-pointer accesses.

The separate QEMU lane
[`b71529835a`](https://github.com/project-starch/capstone-qemu/commit/b71529835a964cf1b268eee6f68518be96a1b2c0)
implements and tests an **isolated U-mode access slice**, without changing
this repository's QEMU submodule pin. It requires the default-off CPU property
`x-capstone-u-mode=true`; this is an experiment selector, not the process ABI.
Without it, existing monitor guests retain CEPC/CTVEC S/U transitions and
ordinary U-mode scalar addressing. With Capstone checks enabled, U-mode
integer, FP, atomic and capability memory accesses reach ordinary Sv39 page
translation and precise M-mode traps. The corrected suite passes **42 checks**:
36 U-mode cases, 4 legacy-path executions and 2 harness controls. It checks
capability types/rights separately from PTE rights, UNINIT STC fault/retry,
and an old/fresh capability pair at one virtual address; the pair uses
debug-minted nodes, not `malloc`/`free`. Six injected source faults are detected.
An existing Linux guest also boots, loads its Capstone module and completes
a shell-command gate with the experiment disabled; no domain application
workload was run for this compatibility check. Source, binary and boot-input
hashes are in the [result record](https://github.com/project-starch/capstone-qemu/blob/b71529835a964cf1b268eee6f68518be96a1b2c0/tests/trusted-linux-u-access/result.json).

**Correction to the initial `db53c89ad0` result:** its reported 23/23 gate
also accepted a deliberate pre-U-mode setup fault because QEMU exited zero.
It missed capability permission/type checks and an UNINIT cursor advance
before a refused PTE store, and the implementation changed existing monitor
return/trap semantics. That result did not establish these contracts. The
corrected runner requires a guest completion marker and rejects setup and
wrong-site faults. UNINIT cursor advancement now follows successful STC writes.
Concurrent PTE changes and two-hart atomicity of the two 64-bit STC writes
remain unqualified. Unsupported vector, Zcmp stack, XThead and cache-block
memory operations are rejected in experimental Capstone U-mode.

The stacked QEMU [physical-tag slice `f38c88108a`](https://github.com/project-starch/capstone-qemu/commit/f38c88108a)
adds a shared map keyed by the TLB's physical 16-byte granule when the
experiment is enabled. Its gate passes **47/47**: the earlier 42 checks plus
physical alias/store clearing, remap identity, and a privileged physical
write, an ordered two-hart tag handoff, and an S-mode store through a second
virtual alias. The prior QEMU binary fails those five new cases. The
[record](https://github.com/project-starch/capstone-qemu/blob/f38c88108a/tests/trusted-linux-u-access/result.json)
contains source and binary hashes. The existing Linux boot/module/shell gate
also passes with the experiment disabled. The parent QEMU pin is unchanged.

This establishes physical tag identity for the directed transitions, not
safe Linux capability processes. The two-hart guest test is an ordered
producer/consumer handoff; concurrent tag/data updates and context transfer
remain unqualified.
Privileged scalar stores clear tags, including the tested S-mode alias store;
full kernel copies, device writes, DMA and frame reclaim still need explicit
contracts and tests. Concurrent PTE
changes can race the post-store privileged physical lookup. The QEMU map has
no migration protocol. A process-specific protected lifetime namespace,
tagged register save/restore, checked syscall buffers with recoverable faults,
and retirement completion also remain open. The next M1 slice should test
tag behavior across a real Linux context change before a process is
used as evidence for the `malloc`/`free` contract.
