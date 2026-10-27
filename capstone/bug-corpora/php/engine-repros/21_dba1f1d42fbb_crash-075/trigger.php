<?php
/* GIT-dba1f1d42fbb — bug #34066 (recursive array_walk causes segfault).
 *
 * php-5.0.0 array_walk (ext/standard/array.c) stores the callback via
 *   zend_get_parameters_ex(argc, &array, &BG(array_walk_func_name), &userdata)
 * so BG(array_walk_func_name) and the userdata slot are POINTERS INTO the
 * growable Zend argument stack. php_array_walk() reads *BG(array_walk_func_name)
 * on every iteration (array.c:1036). If the userland callback pushes enough
 * arguments to force zend_ptr_stack to realloc(), the old stack buffer that the
 * outer array_walk still points into is freed -> the next iteration derefs a
 * dangling pointer => heap-use-after-free. Distinct from the "callback frees the
 * walked array's buckets" class (V5C-002): here the callback never touches the
 * array, it just grows the argument stack.
 *
 * Fixed by storing the callback in a stable C-stack local (tmp) via
 * zend_parse_parameters and setting BG(array_walk_func_name) = &tmp. */

$GLOBALS['a'] = array("k0" => 0, "k1" => 1, "k2" => 2);
$GLOBALS['fired'] = 0;

function many() { return 1; }

function cb(&$v, $k) {
    if (!$GLOBALS['fired']) {
        $GLOBALS['fired'] = 1;
        /* One callback pushes a large argument vector, forcing the Zend
         * argument stack to realloc and freeing the buffer the outer
         * array_walk's callback pointer still references. */
        call_user_func_array('many', array_fill(0, 500000, 7));
    }
}

array_walk($GLOBALS['a'], 'cb');
echo "DONE\n";
