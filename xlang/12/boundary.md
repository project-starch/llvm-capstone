# Language Boundary Violation — mruby #4001 (Row 12)

### Boundary Pointer
The `DATA_PTR` pointer, representing a wrapped C `struct mrb_io` structure containing the open file descriptor state, crosses the language boundary. It is allocated/freed by the C gem `mruby-io` but is owned and stored behind the Ruby `File`/`IO` object's instance variable via `DATA_PTR(self)`.

### Lifetime Violation Details
In the vulnerable implementation of `mrb_io_initialize_copy`:
```c
  fptr_copy = (struct mrb_io *)DATA_PTR(copy);
  if (fptr_copy != NULL) {
    fptr_finalize(mrb, fptr_copy, FALSE);
    mrb_free(mrb, fptr_copy);
  }
  fptr_copy = (struct mrb_io *)mrb_io_alloc(mrb);
  fptr_orig = io_get_open_fptr(mrb, orig);
```
The method first forcefully frees the existing C `mrb_io` structure of the receiver object `copy` (`fptr_copy`) using `mrb_free`. It then calls `io_get_open_fptr` to retrieve the source object's pointer. If the source argument is invalid (e.g. `0` instead of an `IO` object), `io_get_open_fptr` raises a `TypeError` exception.

Since exceptions in mruby execute a longjmp back to Ruby space, execution completely aborts the method before assigning the newly allocated `fptr_copy` back to `DATA_PTR(copy)`. The Ruby `File` object is left in an active state but carries a dangling `DATA_PTR` pointing to the freed C memory. When subsequent Ruby methods (such as `f.close`) are called, the FFI C functions attempt to retrieve and write to this freed structure, producing a heap-use-after-free (UAF) violation.
