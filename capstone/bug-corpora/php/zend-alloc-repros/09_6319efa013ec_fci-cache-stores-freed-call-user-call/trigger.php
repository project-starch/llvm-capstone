<?php
class Faulty {
    function __call($m,$a){ echo "call $m\n"; }
    function NormalMethod($a){ echo "normal\n"; }
}
$f = new Faulty();
$arr = array('x','y');
$r = array_map(array($f,'seg'), $arr);
$r = array_map(array($f,'NormalMethod'), $arr);
