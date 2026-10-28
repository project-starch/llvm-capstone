<?php
// Minimized trigger for heap-use-after-free at zend_execute.c:676 (zend_assign_to_variable, ZE2 php-5.0.0)
// Root cause: by-reference function that returns a PARENTHESIZED reference to an UNDEFINED variable.
// The parens make the return an rvalue/temporary (not a real variable reference); the ZEND_RETURN-by-ref
// + ZEND_ASSIGN_REF path binds $sh as is_ref with a wrong refcount. Rebinding the SAME $sh a second time
// frees the previous reference target while a stale pointer remains reachable. The next array-append
// assignment (ASSIGN_DIM) drives zend_assign_to_variable's is_ref branch, which zval_dtor's the stale
// value (line 612) and then re-reads it via AI_USE_PTR (line 676) -> UAF.
// All three parts are required (verified): parens, the double same-var rebind, the trailing append.
function &f() { return ($sa); }
$sh =& f();
$sh =& f();
$sq[] = 1;
