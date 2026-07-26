# Language Boundary Violation — CVE-2018-10191 (Row 11)

### Boundary Pointer
The environment pointer `e->stack` representing a referenced outer scope stack frame crosses the language boundary. It points to registers holding outer variables on the mruby VM stack.

### Lifetime Violation Details
In deeply nested scope closures (such as 128 nested `instance_eval` blocks), mruby resolves outer variables using the `OP_GETUPVAR` instruction:
```c
struct REnv *e = uvenv(mrb, c);
```
where `c` is the scope level offset. Due to the 8-bit unsigned size of `c`, nesting 128 levels overflows the operand to `0`. This causes `uvenv` to return the innermost scope's local environment instead of the outermost scope.

Because the variable index `b` was compiled to point to a wide outer scope's local register offset (e.g., register `500`), reading `e->stack[b]` from the small innermost stack frame (which only has 2-4 registers allocated) reads outside the allocated environment array, causing an out-of-bounds heap read use-after-free or null-pointer dereference violation.
