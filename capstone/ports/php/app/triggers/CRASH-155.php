class A {}
function deep($n) { if ($n > 0) { deep($n - 1); } return $n; }
function __autoload($c) { deep(800); eval("class $c {}"); }
$o = new A;
var_dump(is_a($o, 'LazyClass'));
var_dump(is_subclass_of($o, 'LazyClass2'));
echo 'done';