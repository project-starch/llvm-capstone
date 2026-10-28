<?php
// Bug #44214: preg_replace_callback + callback that keeps the $match array
// via a global. subpats (refcount 1) is passed by-value to the callback;
// the callback stores it into $array (refcount->2), then the impl does an
// unconditional zval_dtor(subpats)+FREE_ZVAL(subpats), freeing the still-
// referenced array => use-after-free when $array is later read.
$string = 'aaa bbb ccc ddd eee ccc aaa bbb';
$array = array();
function myCallBack($match) {
    global $array;
    $array[] = $match;
    return 'xxx';
}
var_dump(preg_replace_callback('`a+`', 'myCallBack', $string));
var_dump($array);
