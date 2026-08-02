# Language Boundary Violation — CVE-2022-1934

### Boundary Pointer
The `regs` pointer argument in `hash_new_from_values` crosses the language boundary. It points directly to a heap-allocated slice of registers on the mruby VM register stack (`mrb->c->ci->stack`).

### Lifetime Violation Details
When passing keyword arguments to a method call, the interpreter calls the C function `hash_new_from_values(mrb, nk, regs+kidx)` to pack them into a Hash. 

Within `hash_new_from_values`, the loop iterates through the list of registers to construct the hash by calling:
```c
mrb_hash_set(mrb, hash, regs[0], regs[1]);
```
For custom object keys, `mrb_hash_set` invokes key comparison. This executes `obj_eql`, which crosses the language boundary back into Ruby space by calling `bad2.eql?(bad1)`. 

Inside `Bad#eql?`, the Ruby script performs deep recursion, forcing a stack extension that reallocates the register stack and frees the old stack array. When execution returns from the Ruby method back to the C loop in `hash_new_from_values`, the local `regs` pointer is now stale and points to freed memory. During the subsequent loop iteration, accessing `regs[0]` or `regs[1]` reads from freed memory, producing a heap-use-after-free (UAF) violation.
