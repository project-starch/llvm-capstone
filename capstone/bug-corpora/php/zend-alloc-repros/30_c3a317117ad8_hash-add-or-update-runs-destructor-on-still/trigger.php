<?php
/* DEFECT (php-5.0.0, Zend/zend_hash.c:228-233, _zend_hash_add_or_update):
 * the update branch runs ht->pDestructor(p->pData) while bucket p is STILL LINKED
 * in the table, then keeps using p (UPDATE_DATA at :231, *pDest = p->pData at :233).
 * For sizeof(void*) payloads INIT_DATA sets p->pData = &p->pDataPtr, i.e. pData
 * aliases storage INSIDE the bucket. zval_ptr_dtor -> __destruct -> unset($GLOBALS['x'])
 * -> zend_hash_del_key_or_index() efree()s that very bucket; control then returns into
 * _zval_ptr_dtor which dereferences *zval_ptr (== the freed bucket's pDataPtr slot)
 * and free()s whatever pointer now lives there.
 * Reached from userland via extract() (ext/standard/array.c:1389 -> zend_hash_update
 * on EG(active_symbol_table)).
 * ASan: heap-use-after-free READ of size 8 in _zval_ptr_dtor
 *       (Zend/zend_execute_API.c:392) <- _zend_hash_add_or_update (Zend/zend_hash.c:229)
 */
class D { function __destruct() { unset($GLOBALS['x']); } }
$x = new D();
extract(array('x' => 'newval'));
echo "after=", $x, "\n";
