$s = str_repeat('A', 16);
$mask = "A\0";
strspn($s, $mask, 0, 4294967296);
echo 'done';