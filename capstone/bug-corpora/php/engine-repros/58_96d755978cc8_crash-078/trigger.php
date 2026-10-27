<?php
// bug #29253: array_diff/array_intersect family copies its first argument with
// RETVAL_ZVAL(*args[0],1,0). When that argument is $GLOBALS the return_value's
// HashTable aliases &EG(symbol_table) (no symbol-table guard at 5.0.0). The diff
// algorithm then sorts/mutates that shared table while holding Bucket* pointers,
// so the comparator dereferences freed/moved buckets -> heap-use-after-free.
error_reporting(0);
$zz = $GLOBALS;
$gg = 'afad';
var_dump(array_diff_assoc($GLOBALS, $zz));
var_dump($gg);
echo "done\n";
