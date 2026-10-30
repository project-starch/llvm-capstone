<?php
$arr = array();
class A {
    public $x = "data";
    function __construct() {
        global $arr;
        $arr[] = $this;
        throw new Exception("f");
    }
    function __destruct() { echo "dtor\n"; }
}
for ($i=0;$i<3;$i++){
  try { new A(); } catch (Exception $e) {}
}
foreach ($arr as $o) { var_dump($o->x); }
?>
