<?php
class C {
  function __get($name) { $GLOBALS['saved'] = $name; return 1; }
}
$o = new C;
$k = array(1);
$v = $o->{$k};
var_dump($GLOBALS['saved']);
