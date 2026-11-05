<?php
// Bug: cascading assignment of an IS_TMP_VAR into a string offset.
// The temporary's string buffer is freed by the offset-write path, but the
// same (now-freed) tmp zval is reused as the assignment's result value.
$s = "AB";
$x = ($s[1] = chr(67) . "");   // (chr(67)."") is an IS_TMP_VAR
var_dump($x);
