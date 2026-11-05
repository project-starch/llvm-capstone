<?php
// Minimized trigger for zend_execute.c:2778 heap-use-after-free (zend_do_fcall_common_helper).
// __call trampoline path efree()s the temporary overloaded zend_function (EX(fbc)) at line 2767,
// then the post-call This-cleanup at line 2778 dereferences EX(fbc)->common.fn_flags because an
// exception is pending (the error handler converts the E_NOTICE from `$undef` into a thrown
// exception). Reading the freed fbc -> UAF.
function h($n, $m) { throw new Exception; }
set_error_handler("h");
class C { function __call($a, $b) { $undef; } }
$o = new C;
$o->test();
