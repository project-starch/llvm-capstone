<?php
/* php-5.0.0 Zend engine-core: debug_print_backtrace() never resets arg_array
 * to NULL inside its frame-walk loop (Zend/zend_builtin_functions.c:1444).
 * Frame #0 (the user function) builds arg_array, prints it and
 * zval_ptr_dtor()s it.  Frame #1 is the eval() frame: function_name==NULL and
 * ZEND_EVAL sets build_filename_arg=0, so arg_array is NOT reassigned.
 * The `if (arg_array)` test then still sees the freed pointer ->
 * debug_print_backtrace_args() reads the freed zval (heap-use-after-free at
 * Zend/zend_builtin_functions.c:1433) and zval_ptr_dtor() frees it a 2nd time.
 * Fixed upstream by commit 1088e28dfa6a ("Fix debug_trace with eval"),
 * which adds `arg_array = NULL;` at the top of the loop.
 */
function f($a, $b, $c) {
    debug_print_backtrace();
}
eval('f(1,2,3);');
