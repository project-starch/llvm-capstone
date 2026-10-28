<?php
class D {
  public $n;
  function __construct($n){ $this->n = $n; }
  function __destruct(){ if ($this->n == 1) { $bt = debug_backtrace(); } }
}
function f($a, $b) { return 1; }
function g() { f(new D(1), new D(2)); }
function outer() { g(); }
outer();
echo "done\n";
