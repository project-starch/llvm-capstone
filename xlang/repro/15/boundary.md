# Language Boundary Violation — mruby #3722 (Row 15)

### Boundary Pointer
The method argument array pointer `argv` in `mrb_str_format` (`sprintf`) crosses the language boundary. It is passed from mruby's VM execution stack into the C library function `mrb_str_format`.

### Lifetime Violation Details
When retrieving variable arguments `*` in the C code, `mrb_get_args` extracts them as a pointer pointing directly into the active VM register stack:
```c
*var = ARGV + arg_i;
```
Thus, the local `argv` pointer in `mrb_str_format` points directly into the active stack of registers.

During the execution of `mrb_str_format` to format the format string (e.g. `sprintf("%s %s", bad, "extra")`), the function iterates over the format arguments. When it formats `bad` via `%s`, it calls `mrb_obj_as_string`, which crosses the language boundary back into Ruby space to execute the object's custom `to_s` method `Bad#to_s`.

Within `Bad#to_s`, the Ruby script performs a deep recursive call that triggers a VM stack extension. The VM reallocates the register stack to a new heap location and frees the old stack array. When the Ruby execution returns, the local `argv` pointer in `mrb_str_format` is now stale. In the subsequent iteration of formatting `"extra"`, accessing `argv[1]` from this stale pointer results in a heap read use-after-free (UAF) violation.
