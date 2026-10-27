<?php
// bug #34277: array_filter holds zval** (input/callback) into the argument
// stack across a userland callback. A deeply nested call inside the callback
// reallocates (moves) the argument stack, dangling input/callback; the next
// loop iteration reads Z_ARRVAL_PP(input) -> use-after-realloc read.
function deep($n) {
    if ($n > 0) { return deep($n - 1); }
    return 1;
}
function cb($x) {
    deep(4000);   // grow+realloc the argument stack while in the callback
    return true;
}
$a = array();
for ($i = 0; $i < 50; $i++) { $a[$i] = $i; }
array_filter($a, 'cb');
echo "done\n";
