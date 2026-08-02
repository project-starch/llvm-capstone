# CVE-2025-13120 / mruby #6649 — use-after-realloc in Array#sort!
#
# mrb_ary_sort_bang's heap sort captures the array's raw data pointer once
# (`a` in heapify/insertion_sort) and keeps using it for element access. The
# comparator is a RUBY BLOCK, so it can re-enter the interpreter and mutate the
# array: slice! shrinks 100 elements to 2, which reallocates the backing store
# and frees the old buffer. The sort then keeps indexing through the stale
# pointer.
#
# The pre-fix guard in sort_cmp only tests `RARRAY_PTR(ary) != p`, which catches
# a move but not a length change; the fix (eb398971) re-reads RARRAY_PTR inside
# sort_cmp and adds `RARRAY_LEN(ary) != n`.
#
# Must exceed SMALL_ARRAY_SORT_THRESHOLD, or insertion sort is used and the
# heap-sort path that caches the pointer is never entered.
#
# Upstream reporter's input, unmodified.
a = (0..99).to_a
a.sort! { |x, y| a.slice!(1, 98); (x < y) ? -1 : 1 }
puts "NO-FAULT: sort! completed, len=#{a.length}"
