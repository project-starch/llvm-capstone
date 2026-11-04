# Checked kernel copies: experimental interface

The checked-copy interface is a default-off native-task prototype. The current
runtime continues to use its held exchange buffer. A capability argument is
`cn_capref` from the protected context: context identity, escape epoch and
register slot. A scalar cursor or LCC metadata cannot supply authority.

`cu_copy` takes that reference, an unsigned byte offset, a trusted kernel
buffer, length and direction, and returns bytes completed plus a capability
cause. The shared loop authorizes each page-bounded chunk, drains before
leaving exclusion, resolves partial faults outside the window and rechecks
the original reference on every retry. A persistent fault has a bounded retry
budget. Zero-length operations authenticate the capability but need no
mapping hold or memory access. Unsupported operations fail closed.

The Linux adapter is `module/checked-copy-dom.h`. Its internal
`cv_checked_copy` entry requires `vm_lock`, a retained invocation reference,
a kernel-resident buffer, the owning native main thread, a protected kernel
frame and exactly one possible CPU. It retains an additional file reference
across sleeping callbacks. A chunk holds mmap_read_lock, disables preemption,
interrupts and page faults, executes CSCHECKR/W, uses `__copy_*_inatomic`,
drains scalar stores and then leaves the window. Anonymous private faults use
`handle_mm_fault` without a dummy store. File/device mappings are unsupported.

Pinned mode preserves page admission and rejects changed backing. A failure
after fault handling that cannot restore the accounting makes the context
terminal, preventing tagged stores to an untracked resident page. This does
not qualify tag-preserving guest COW. The unpinned kernel's page lifecycle
and the native context/grant integrations need separate qualification.

A held native unmap retires its node immediately and returns success. Its
arena remains in the mapping trees, and its VMA reserves the address, until
the last hold releases it. Physical retirement then clears/releases pinned
pages and performs the deferred unmap. Remap and protection changes refuse
held/retired arenas in this initial whole-grant subset. Thread-owned frame
and initial transport mappings retain their existing retirement restrictions.

The QEMU prototype binds checks to its hidden immutable escaped registers,
selected namespace/satp and a fresh escape epoch at frame byte 656. This is
an experimental adapter to the [context proposal](native-context.md), not
lossless physical-slot virtualization. Its six-word physical request layout
and tests are documented in
[the processor experiment](../../../capstone-qemu/tests/capstone-checked-copy/README.md).

**Integration is unresolved.** The frozen custom-0/funct3=6/funct7=0,1
encodings overlap existing CPTAG immediates. QEMU refuses the checked-copy
and raw-moves properties together; the contract owner must allocate disjoint
encodings before integrating raw COW and checked copies. No syscall/SDK
consumer calls the module helper yet. Default `checked_copy=false` remains.

The [hosted experiment](../../../runtime/virtual/tests/checked-copy/README.md)
links the shared copy loop to the native context double and checks forced
thread-switch object/mapping reuse and fault retries with unsafe controls.
Those results do not establish real kernel scheduling, guest COW, instruction
attempt budgets, the native quick/full census or the nine landing gates.
All remain required before enabling or landing the native path.
