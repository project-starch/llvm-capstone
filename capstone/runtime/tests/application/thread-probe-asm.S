/* Probe-only helper for thread-probe.c (docs/plans/delegation-threads.md,
 * Probe B, domain phase). Not part of the runtime. */
.text

/* void probe_fault_store(volatile int *where): one store at a named
 * instruction, so that the fault a further context takes there can be
 * located in the fault record. */
  .global probe_fault_store
probe_fault_store:
  .global probe_fault_store_insn
probe_fault_store_insn:
  sw zero, 0(a0)
  ret
