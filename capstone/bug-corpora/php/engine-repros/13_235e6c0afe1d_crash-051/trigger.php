<?php
class C {
  function __call($name, $args) {
    $GLOBALS['saved'] = $args;
    $GLOBALS['savedname'] = $name;
  }
}
$c = new C();
$c->foo(array(1,2,3), "hello", 42);
var_dump($GLOBALS['saved']);
var_dump($GLOBALS['savedname']);
