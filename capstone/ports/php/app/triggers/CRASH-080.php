function rd($carry, $item) { return $carry; }
$r = array_reduce(array(1,2,3), 'rd', array('initialarray'));
var_dump($r);