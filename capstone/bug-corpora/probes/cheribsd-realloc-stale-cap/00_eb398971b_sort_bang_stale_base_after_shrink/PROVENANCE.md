# sort_bang_stale_base_after_shrink

Upstream fix `eb398971b` (mruby, 2025-10-27), *"array.c: fix use-after-realloc in Array#sort!;
fix #6649"*. An ancestor of the pin `4.0.0-rc2` (`9d523e2f74f2`), so the defect is fixed there and
this case reconstructs the pre-fix allocator shape.

## The upstream hunk (src/array.c)

```c
-sort_cmp(mrb_state *mrb, mrb_value ary, mrb_value *p, mrb_value a_val, mrb_value b_val, mrb_value blk)
+sort_cmp(mrb_state *mrb, mrb_value ary, mrb_value a_val, mrb_value b_val, mrb_value blk)
 {
+  mrb_value *p = RARRAY_PTR(ary);
+  mrb_int n = RARRAY_LEN(ary);
   ...
-  if (RARRAY_PTR(ary) != p) {
+  if (RARRAY_PTR(ary) != p || RARRAY_LEN(ary) != n) {
     mrb_raise(mrb, E_RUNTIME_ERROR, "array modified during sort");
   }
```

The message states the mechanism: *"when realloc() shrinks an array in-place, it may return the
same pointer, defeating the pointer-only check."* A comparison block that shrinks the array reaches
`ary_shrink_capa`, which halves the capacity via `mrb_realloc`; when that keeps the address the old
guard misses it and the sort reads the captured base in the trimmed tail.

Real: the capture, the shrinking `mrb_realloc`, the stale read. Reduced: 64 entries shrunk to 32,
read at the old last index, as a standalone `case.c`. The measured instrument is the sibling
`probe.c` / `probe-capstone.c`, scenario `mruby-sort-bang-shrink`. The commit subject names the
maintainer's handle in its trailer; cited by hash and path only.
