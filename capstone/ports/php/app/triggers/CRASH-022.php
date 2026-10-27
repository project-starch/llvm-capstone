function cmp($x, $y) { global $a;
  unset($a[2]); unset($a[3]); unset($a[4]); unset($a[5]); unset($a[6]);
  $a[100]=1; $a[101]=2; $a[102]=3; $a[103]=4; $a[104]=5;
  return ($x < $y) ? -1 : 1; }
$a = array(5,3,8,1,9,2,7,4,6,0,11,12,13,14,15,16,17,18,19,20);
usort($a, 'cmp');
echo 'done ', count($a);