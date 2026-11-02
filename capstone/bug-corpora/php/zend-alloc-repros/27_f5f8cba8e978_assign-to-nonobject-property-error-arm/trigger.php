<?php
/* Re-audit of GIT f5f8cba8e978 (bug #44660, "assignment to property of non-object").
 *
 * php-5.0.0 Zend/zend_execute.c:363 zend_assign_to_object(), non-object error arm:
 *
 *     value = get_zval_ptr(value_op, Ts, &free_value, BP_VAR_R);   // :367  -> already PZVAL_UNLOCKs an IS_VAR
 *     ...
 *     if (object->type != IS_OBJECT) {
 *         zend_error(E_WARNING, "Attempt to assign property of non-object");   // :375
 *         FREE_OP(Ts, op2, EG(free_op2));
 *         *retval = EG(uninitialized_zval_ptr);
 *         SELECTIVE_PZVAL_LOCK(*retval, result);
 *         PZVAL_UNLOCK(value);                                     // :380  <-- SECOND unlock
 *         return;
 *     }
 *
 * _get_zval_ptr() (zend_execute.c:98-102) already released the FETCH lock for an
 * IS_VAR operand, so :380 decrements the value zval a second time.  refcount hits 0,
 * zend_pzval_unlock_func pushes it into EG(garbage), and zend_clean_garbage frees it
 * while $s in the symbol table still points at it -> dangling symbol-table entry.
 * The sibling compound-assign path (zend_binary_assign_op_obj_helper, :1636-1638)
 * uses the CORRECT `FREE_OP(Ts, value, free_value)` here, which is a no-op for IS_VAR.
 *
 * Requires a *variable* (IS_VAR) value operand: `$a->p = 1;` (IS_CONST) does not fault.
 */
$s = str_repeat("A", 8);
$a = true;
$a->p = $s;                 /* frees $s's zval while the symbol table still holds it */
$q1 = array(1,2,3); $q2 = "XXXXXXXXXXXXXXXX"; $q3 = 1.5;   /* churn: reuse the freed slot */
echo $s, "\n";              /* ZEND_FETCH_R -> zend_pzval_lock_func on freed zval */
var_dump($s);
