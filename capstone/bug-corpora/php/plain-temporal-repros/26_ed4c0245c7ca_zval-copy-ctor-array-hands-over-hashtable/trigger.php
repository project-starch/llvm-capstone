<?php
/* php-5.0.0 Zend/zend_variables.c _zval_copy_ctor(), case IS_ARRAY:
 *
 *     ALLOC_HASHTABLE_REL(tmp_ht);
 *     zend_hash_init(tmp_ht, 0, NULL, ZVAL_PTR_DTOR, 0);
 *     zend_hash_copy(tmp_ht, original_ht, zval_add_ref, &tmp, sizeof(zval*));
 *     zvalue->value.ht = tmp_ht;      <-- ownership handover happens LAST
 *
 * `zvalue` is ALREADY reachable (SEPARATE_ZVAL stores the new zval into the
 * symbol-table slot BEFORE calling zval_copy_ctor), yet it still points at
 * original_ht for the whole duration of zend_hash_copy().  Any zend_bailout()
 * raised inside that window (here: the SIGPROF max_execution_time handler,
 * Zend/zend_execute_API.c zend_timeout -> zend_error(E_ERROR) -> longjmp)
 * leaves TWO independent zvals ($a and $b) each owning the SAME HashTable.
 * shutdown_executor() then destroys the symbol table and calls
 * zend_hash_destroy()+FREE_HASHTABLE() on that one HashTable twice.
 *
 * Upstream fix (9dfa843a386 / merges ed4c0245c7ca, e120438c8c06,
 * "Partial fix for bug #68365 - zend_mm_heap corrupted after memory overflow
 * in zend_hash_copy") simply moves `zvalue->value.ht = tmp_ht;` BEFORE the
 * zend_hash_copy() call.
 */
$a = array_fill(0, 5000000, 1);   /* big enough that one copy costs > 1s CPU */
$b = $a;                          /* refcount 2, shared ht, no copy yet */
set_time_limit(1);                /* arms ITIMER_PROF for 1s of CPU */
$t = microtime(true);             /* burn ~0.45s so the alarm lands mid-copy, */
while (microtime(true) - $t < 0.45) { }   /* not just after it (robustness only) */
$b[0] = 1;   /* FETCH_DIM_W -> SEPARATE_ZVAL -> zval_copy_ctor -> zend_hash_copy -> BAILOUT
              * -> $a and $b now BOTH own the same HashTable -> double zend_hash_destroy
              *    + double FREE_HASHTABLE at shutdown_executor(). */
echo "no bailout\n";
