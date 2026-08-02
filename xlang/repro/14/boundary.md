# Language Boundary Violation — mruby #3596 (Row 14)

### Boundary Pointer
The `stbase` stack buffer pointer representing the active VM context stack (`mrb->c->stbase`) crosses the language boundary. It points to a heap-allocated array of `mrb_value` registers where the virtual machine stores local variables, temporary values, and method parameters.

### Lifetime Violation Details
In mruby's conservative garbage collection stack-root scanner, the function `mark_context_stack` scans the active registers on the stack to mark alive objects and protect them from garbage collection:
```c
  e = c->stack - c->stbase;
  if (c->ci) e += c->ci->nregs;
```
When a method returns or execution exits a block, the stack pointer `c->stack` and the callinfo register count `c->ci->nregs` shrink (decrement). However, in the vulnerable version, the unused registers above the new stack limit `e` (up to `c->stend`) are NOT cleared and continue to carry raw pointers to recently returned, now discarded objects.

Since these discarded temporary objects are no longer referenced anywhere else, a subsequent garbage collection sweep forcefully frees them. This leaves the inactive stack region containing stale pointers pointing to deallocated heap blocks. 

If execution subsequently invokes a new method, the stack pointer `c->stack` expands over this unused region again, incorporating the stale registers into the active stack range `e`. During the subsequent GC marking phase, `mark_context_stack` scans the stack up to the new expanded limit `e`, reads the uninitialized stale pointer pointing to the freed object, and attempts to mark it. This produces a heap Use-After-Free (UAF) violation.
