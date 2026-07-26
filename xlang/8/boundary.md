# Language Boundary Violation — mruby #4926 (Row 8)

### Boundary Pointer
The method argument array pointer `argv` in `hash_values_at` crosses the language boundary. It is passed from mruby's VM execution stack into the C extension method.

### Lifetime Violation Details
Due to a logical inversion bug in `mrb_get_args()`'s argument copying condition:
```c
mrb_bool nocopy = altmode || argv_on_stack ? TRUE : FALSE;
```
when the arguments are on the stack (`argv_on_stack` is true), mruby does NOT copy the arguments array off the stack, leaving `argv` pointing directly into the active VM register stack (`mrb->c->ci->stack`).

When `values_at` is called (e.g. `c.values_at(bad1, bad2, bad3)`), the C function iterates over `argv` to look up each key in the hash:
```c
for (i=0; i<argc; i++) {
  mrb_value key = argv[i];
  ...
}
```
During the second lookup `argv[1]` (`bad2`), `mrb_hash_get` is called. For custom keys, this invokes the key's `hash` method, crossing the language boundary back to Ruby space to execute `Bad#hash`.

Inside `Bad#hash`, the Ruby script performs a deep recursive call that triggers a VM stack extension. The VM reallocates the register stack and frees the old stack array. When the Ruby execution returns, the local `argv` pointer in `hash_values_at` is now stale. In the subsequent iteration (`argv[2]`), accessing the stale pointer to retrieve `bad3` results in a heap read use-after-free (UAF).
