# Language Boundary Violation — mruby #3829 (Row 9)

### Boundary Pointer
The `ptr` pointer representing string bytes of dynamically evaluated `irep` string literal pools crosses the language boundary. It is allocated as part of the bytecode's literal pool but is read/referenced directly by the Ruby string object `RString`.

### Lifetime Violation Details
When literal strings are evaluated in Ruby (e.g., dynamically compiled via `eval`), the interpreter creates a frozen pool string representing the literal, stored inside the dynamically allocated `mrb_irep`'s `pool` array. When a substring is taken (such as `str[1..-2]`), the function `byte_subseq` creates a shared string (`MRB_STR_FSHARED`) that points directly to the C pointer `orig->as.heap.ptr` of the pool string.

However, because the `RString` substring object does not hold a garbage collection reference count to the underlying `mrb_irep` structure, the garbage collector remains unaware of this pointer cross-reference. When the evaluated dynamic proc is swept and garbage collected, it triggers `mrb_irep_decref()` which decrements the `irep`'s reference count. When the count drops to 0, `mrb_irep_free` is called, which forcefully frees all string literal arrays inside the `pool` array.

This leaves the active Ruby substring (`$sub`) pointing directly to the freed data. During the subsequent garbage collection marking phase (`mrb_gc_mark()`) or string printing, the interpreter reads from this stale pointer, resulting in a heap-use-after-free (UAF) violation.
