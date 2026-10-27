<?php
$sum = 0;
for ($i = 0; $i < 2000; $i++) {
  $a = array(); for ($j = 0; $j < 20; $j++) { $a[] = str_repeat(chr(65 + $j), $i % 7 + 1); }
  $s = implode("-", $a); $sum += strlen($s); unset($a[3]); unset($a[10]); $a = null;
}
echo "churn ", $sum, "\n";
