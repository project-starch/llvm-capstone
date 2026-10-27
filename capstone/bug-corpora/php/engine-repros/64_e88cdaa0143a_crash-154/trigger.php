<?php
/* GIT-e88cdaa0143a -- php-5.0.0 memory-safety fault
 * Root cause (engine-core): Zend/zend_variables.c:47-55, _zval_dtor(), IS_ARRAY arm:
 *     case IS_ARRAY: case IS_CONSTANT_ARRAY: {
 *         if (zvalue->value.ht && (zvalue->value.ht != &EG(symbol_table))) {
 *             zend_hash_destroy(zvalue->value.ht);      <- runs userland __destruct()
 *             FREE_HASHTABLE(zvalue->value.ht);
 *         }
 *     }
 * The zval keeps Z_TYPE == IS_ARRAY and a live value.ht for the whole duration of
 * zend_hash_destroy().  A destructor that re-enters the engine and destroys the SAME
 * zval (here: writing the still-installed global slot that is being torn down) therefore
 * enters zend_hash_destroy() on the SAME HashTable a second time; the inner pass
 * pefree()s the buckets that the outer pass is still walking -> heap-use-after-free.
 * Upstream fix e88cdaa0143a adds `Z_TYPE_P(zvalue) = IS_NULL;` immediately before
 * zend_hash_destroy() so the re-entrant destruction becomes a no-op ("break possible cycles").
 */
class D { function __destruct() { $GLOBALS['a'] = 2; } }
$a = array(new D(), 'x', 'y');
$a = 1;
echo "after\n";
