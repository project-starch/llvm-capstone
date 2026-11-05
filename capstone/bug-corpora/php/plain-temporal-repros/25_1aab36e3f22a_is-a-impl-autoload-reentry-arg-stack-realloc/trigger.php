<?php
/* php-5.0.0 -- Zend/zend_builtin_functions.c is_a_impl()
 *
 * is_a()/is_subclass_of() fetch their arguments with zend_get_parameters_ex(),
 * which hands back zval** pointers that point DIRECTLY INTO the
 * EG(argument_stack).elements block.  is_a_impl() then calls
 * zend_lookup_class(), which at 5.0.0 runs userland __autoload().  Any nested
 * userland call deep enough to make zend_ptr_stack_n_push() grow the argument
 * stack erealloc()s (== frees) that block, so `obj` dangles.  On the
 * lookup-SUCCESS path is_a_impl() then dereferences it:
 *     line 621:  instance_ce = Z_OBJCE_PP(obj);          (is_a)
 *     line 619:  instance_ce = Z_OBJCE_PP(obj)->parent;  (is_subclass_of)
 * -> heap-use-after-free READ.
 *
 * Fixed by 1aab36e3f22a ("is_a() and is_subclass_of() should NOT call
 * autoload"), which removes the userland re-entry entirely.
 *
 * Counterfactual: growing the stack BEFORE the call (so __autoload does not
 * have to grow it) or pre-declaring the class runs clean.
 */
class A {}

function deep($n) { if ($n > 0) { deep($n - 1); } return $n; }

function __autoload($c) {
    deep(800);            /* erealloc()s EG(argument_stack).elements */
    eval("class $c {}");  /* make zend_lookup_class() SUCCEED so obj is used */
}

$o = new A;
var_dump(is_a($o, 'LazyClass'));           /* UAF at is_a_impl:621 */
var_dump(is_subclass_of($o, 'LazyClass2'));/* UAF at is_a_impl:619 */
echo "done\n";
