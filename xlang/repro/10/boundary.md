# Language Boundary Violation — CVE-2022-1106

### Boundary Pointer
The `regs` pointer, representing the mruby VM register stack (`mrb->c->ci->stack`), crosses the language boundary. It points to a heap-allocated array of `mrb_value` objects where VM local registers are stored.

### Lifetime Violation Details
During the evaluation of the VM instruction `OP_RANGE_INC`, the interpreter evaluates the expression:
```c
regs[a] = mrb_range_new(mrb, regs[a], regs[a+1], FALSE);
```
Since `regs` is a preprocessor macro expanding to `(mrb->c->ci->stack)`, the compiler evaluates the destination address of the LHS assignment `&(regs[a])` before or during the RHS function call. While `mrb_range_new()` is executing, it invokes `mrb_cmp()`, which crosses the language boundary back to Ruby space to execute a user-defined comparison method `Bad#<=>`. 

Within `Bad#<=>`, the Ruby script performs a recursive call that extends the VM stack beyond its current capacity. The VM allocates a new stack array and frees the old one via `mrb_realloc_simple()`. When the VM execution returns and `mrb_range_new()` returns, the C compiler writes the returned range object to the cached destination address `&(regs[a])` on the old stack, which has already been freed. This results in a heap write use-after-free (UAF) of size 8 on the host machine.
