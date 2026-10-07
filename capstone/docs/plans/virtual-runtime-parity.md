# Virtual runtime replacement gate

The replacement target is the current one-hart physical application runtime,
not new threads, fork, swap or file-backed mappings. Keep the physical path as
a comparison until the virtual path passes its functional and safety contracts.
Work lives on `virtual-capstone-runtime-parity`.

1. Reuse launcher process services, signal handling, structured fault records,
   real signal termination, argv/binfmt conventions and host CLI transport.
   Keep private descriptors out of delegated calls and spawned children.
2. Restore allocation-sized bounds and qualify the existing heap and delegated
   buffer contracts. Include small/large overflows, stale aliases, same-address
   reuse, double free, neighbouring live objects and realloc of tagged data.
   Require operation markers and exact fault sites; an earlier setup fault is
   not a safety pass. Run an explicitly unsafe comparison as a detector control.
3. Reclaim invalid lifetime IDs only after all stale tags in the owning
   namespace's memory and saved contexts are cleared. Retain dead PCC identities.
   Qualify more than 200,000 allocation/free cycles, namespace separation,
   retained stale pointers and genuine live-node exhaustion.
4. Restore existing local mapping/SysV compatibility and allocator grants used
   by nested ports. These remain process-local compatibility, not shared tagged
   memory between independent Linux processes.
5. Run the existing process, signal, socket, buffer and application fixtures
   through the virtual launcher; broaden the rebuilt port matrix. Record
   reproducible inputs and explicit residuals before making it the default.

Processor tests must continue to cover capability/PTE rights separately,
revoked PCC, untagged accesses, linear-transfer fault atomicity and retry,
forbidden privilege operations and restoration of the trusted caller.
Compact-bounds behavior and QEMU-only collection must not be reported as RTL
qualification. Removing the historical physical implementation is a separate
cleanup after the replacement gate, not a prerequisite for comparison.
