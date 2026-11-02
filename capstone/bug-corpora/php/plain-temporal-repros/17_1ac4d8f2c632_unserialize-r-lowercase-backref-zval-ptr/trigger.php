<?php
// Bug #65481 root cause: var_unserializer 'r:' reference handler frees the
// current *rval via zval_ptr_dtor while that zval is still tracked in the
// unserialize var_hash, leaving a dangling entry that a later back-reference
// dereferences -> heap-use-after-free.
$x = unserialize('a:2:{i:0;r:2;i:1;r:2;}');
var_dump($x);
