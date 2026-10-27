class D { function __destruct() { unset($GLOBALS['x']); } }
$x = new D();
extract(array('x' => 'newval'));
echo 'after=', $x;