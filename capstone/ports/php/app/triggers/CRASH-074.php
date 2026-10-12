function cb($value, $key, &$userdata) { $userdata = str_repeat('A', 32); }
$array = array('hello', array('world', array('deep')));
$userdata = 'x';
array_walk_recursive($array, 'cb', $userdata);
echo 'Done';