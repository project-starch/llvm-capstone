# Language Boundary Violation — CVE-2022-1071

### Boundary Pointer
The `regs` pointer, representing the mruby VM register stack (`mrb->c->ci->stack`), crosses the language boundary. It points to a heap-allocated array of `mrb_value` objects where VM local registers are stored.

### Lifetime Violation Details
During the evaluation of the VM instruction `OP_GETCONST`, the interpreter evaluates the expression:
```c
regs[a] = mrb_vm_const_get(mrb, syms[b]);
```
Since `regs` is a preprocessor macro expanding to `(mrb->c->ci->stack)`, the compiler evaluates the destination address of the LHS assignment `&(regs[a])` before or during the RHS function call. While `mrb_vm_const_get()` is executing, it triggers a constant lookup. If the constant is missing, it invokes `const_missing`, which crosses the language boundary back to Ruby space to execute a user-defined method `M.const_missing`.

Within `const_missing`, the Ruby script performs a recursive call that extends the VM stack beyond its current capacity. The VM allocates a new stack array and frees the old one via `mrb_realloc_simple()`. When the VM execution returns and `mrb_vm_const_get()` returns, the C compiler writes the returned range/const object to the cached destination address `&(regs[a])` on the old stack, which has already been freed. This results in a heap write use-after-free (UAF) of size 8 on the host machine.
