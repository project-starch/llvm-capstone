<?php
/* GIT-86434be9462c -- php-5.0.0 memory-safety fault
 * Root cause (engine-core): Zend/zend_hash.c:489-497, zend_hash_del_key_or_index().
 * The bucket is unlinked from arBuckets/pListHead/pListTail FIRST, then
 *     if (ht->pDestructor) { ht->pDestructor(p->pData); }   <- userland __destruct() re-enters here
 *     ...
 *     ht->nNumOfElements--;                                 <- decremented only AFTER the dtor
 * so any code that runs inside the element destructor observes
 * nNumOfElements == (real number of buckets) + 1.
 *
 * Consequence: an internal function that gates on zend_hash_num_elements() and then
 * blindly trusts zend_hash_get_current_data() dereferences an UNINITIALISED pointer.
 * _phpi_pop() (ext/standard/array.c:1884-1894) is exactly that shape:
 *     if (zend_hash_num_elements(ht) == 0) return;      <- sees the stale 1, passes
 *     zend_hash_internal_pointer_end(ht);
 *     zend_hash_get_current_data(ht, (void**)&val);     <- FAILS, leaves `val` uninitialised
 *     *return_value = **val;                            <- wild pointer deref -> SEGV
 * Upstream fix 86434be9462c moves `ht->nNumOfElements--` above the pDestructor call.
 */
class D {
    function __destruct() {
        /* the array now has ZERO live buckets but still reports count()==1 */
        var_dump(array_pop($GLOBALS['a']));
    }
}
$a = array(new D());
unset($a[0]);
echo "after\n";
