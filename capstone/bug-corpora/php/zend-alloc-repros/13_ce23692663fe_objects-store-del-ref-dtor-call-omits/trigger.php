<?php
class test {
  public $c = 1;
  function __destruct() {
    $GLOBALS['p']->c++;
    print $GLOBALS['p']->c."\n";
    var_dump($GLOBALS['p']);
  }
}
$p = new test;
$p = null;
?>
