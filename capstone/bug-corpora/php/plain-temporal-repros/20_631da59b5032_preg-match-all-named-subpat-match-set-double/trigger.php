<?php
/* preg_match_all PREG_PATTERN_ORDER + named capture group: match_sets[i]
   inserted twice (by name and by index) with refcount 1, missing ZVAL_ADDREF
   => double-free / use-after-free when the result array is destroyed. */
function f() {
    $s = 'what the word and the other word the';
    preg_match_all('/(?P<word>the)/', $s, $m);
    return $m['word'];
}
$w = f();
var_dump($w);
