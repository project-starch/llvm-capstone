# hvaux_interior_after_hsplit

Upstream fix `4547997058` (Perl, 2014-03-04), *"[perl #121362] overload optimisation added a
SEGV"*. Predates the pin `5.36.3` and is an ancestor of it, so the defect is fixed there and this
case reconstructs the pre-fix allocator shape.

## The upstream hunk (gv.c, Perl_Gv_AMupdate)

```c
     if (!deref_seen)
-        /* none of @{} etc overloaded; we can do $obj->[N] quicker */
-        aux->xhv_aux_flags |= HvAUXf_NO_DEREF;
+        /* none of @{} etc overloaded; we can do $obj->[N] quicker.
+         * NB - aux var invalid here, HvARRAY() could have been
+         * reallocated since it was assigned to */
+        HvAUX(stash)->xhv_aux_flags |= HvAUXf_NO_DEREF;
```

The message states the mechanism: *"the 'aux' variable is set to HvAUX(hv) ... by then,
S_hsplit() may have been called, and thus HvARRAY (and HvAUX()) may have been reallocated."* The
xpvhv_aux struct sits at the tail of the bucket array, so `aux` is an interior pointer into that
one allocation; `S_hsplit` Renews the array and the pre-fix write through the stale `aux` lands in
live buckets. The fix re-reads `HvAUX(stash)`.

Real: the interior capture, the growing `Renew`, the stale interior write. Reduced: a 200-byte
allocation with an interior pointer at 192 reallocated to 224 (one jemalloc small size class, so it
can stay in place), as a standalone `case.c`. The measured instrument is the sibling `probe.c` /
`probe-capstone.c`, scenario `perl-hsplit-aux-grow`. The commit subject names the author; cited by
hash and path only.
