<?php
// GIT-9a98904ddd0e -- Zend/zend_API.c:1955 zend_read_property()
//
// At php-5.0.0 zend_read_property() builds the member zval on the C stack and
// never refcount-initialises it, and points it at a *static* C string literal
// with dup=0:
//
//   zval property, *value;                         /* zend_API.c:1957 */
//   ZVAL_STRINGL(&property, name, name_length, 0);  /* zend_API.c:1965 */
//   value = Z_OBJ_HT_P(object)->read_property(object, &property, silent);
//
// (no MAKE_STD_ZVAL / INIT_PZVAL, no zval_ptr_dtor -- and `silent` is passed
//  where the handler expects a BP_VAR_* `type`, which is what 9a98904ddd0e
//  finally repaired together with the heap-allocated member zval.)
//
// When the requested property is absent and the class has __get(),
// zend_std_read_property() forwards that stack zval straight into
// zend_std_call_getter() -> call_user_function_ex(), which refcount-manages it
// as a normal userland argument and finally zval_ptr_dtor()s it in
// zend_ptr_stack_clear_multiple().  That efree()s the .rodata literal
// "message" belonging to Zend/zend_exceptions.c.
//
// Exception::getMessage() reaches zend_read_property() from C, so unsetting the
// protected `message` property of an Exception subclass that declares __get()
// is enough.  No odd syntax, no non-string member (so this is NOT the
// zend_std_read_property() tmp_member path).
class E3 extends Exception {
  function __construct(){ parent::__construct("m"); unset($this->message); }
  function __get($n){ return "v"; }
}
$e = new E3();
echo $e->getMessage(), "\n";
echo "end\n";
