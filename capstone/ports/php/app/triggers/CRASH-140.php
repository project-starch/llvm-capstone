function f($a, $b, $c) { debug_print_backtrace(); }
eval('f(1,2,3);');