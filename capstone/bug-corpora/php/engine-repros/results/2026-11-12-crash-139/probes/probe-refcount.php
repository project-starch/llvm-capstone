<?php
// probe-refcount.php -- CRASH-139's trigger with one change: __get() prints the reference count
// of the zval it was handed (debug_zval_dump), which is the member zval zend_read_property()
// built on the C stack without initialising that field. Everything before __get() runs is the
// trigger's, statement for statement, so the field is laid down the same way.
class E3 extends Exception {
  function __construct(){ parent::__construct("m"); unset($this->message); }
  function __get($n){ debug_zval_dump($n); return "v"; }
}
$e = new E3();
echo $e->getMessage(), "\n";
echo "end\n";
