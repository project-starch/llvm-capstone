<?php
ini_set('zend.ze1_compatibility_mode', 1);
class test { }
$t = new test;
$t = $t; // self-assignment: refcount-- frees object, then reads/clones freed zval
var_dump($t);
