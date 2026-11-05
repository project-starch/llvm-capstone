<?php
if (defined("pass3")) {
  class ErrorClass { }
} else if (defined("pass2")) {
  class TestClass {
    function __construct() { }
    function TestClass() { $this->__construct(); }
  }
} else {
  function errorHandler($n, $m, $f, $l) {
    define("pass3", 1);
    include(__FILE__);
    die("Error: $m\n");
  }
  set_error_handler('errorHandler');
  define("pass2", 1);
  include(__FILE__);
}
