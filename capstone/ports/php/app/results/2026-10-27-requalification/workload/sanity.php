<?php
// revocation-sanity: a benign workload. Every arm must run this to its normal end;
// an arm that cannot is not a column, and a silence on it is not a MISSED.
// It sorts, three ways, with a comparator that touches nothing: PHP's own
// _zend_qsort_swap moved Bucket* elements through int and char stores, which
// strips their tags, so every sort faulted at its first comparison (patch 0002).
// Before this control existed, CRASH-022 and CRASH-078 read that fault as a catch.
function phpb_cmp($x, $y) { return ($x < $y) ? -1 : 1; }
$k = array(3 => 'c', 1 => 'a', 2 => 'b'); ksort($k);
$w = array(5, 3, 8, 1, 9, 2, 7, 4, 6, 0); sort($w); usort($w, 'phpb_cmp');
$a = array();
for ($i = 0; $i < 400; $i++) { $a[] = str_repeat('x', ($i % 23) + 1); }
for ($i = 0; $i < 400; $i += 2) { unset($a[$i]); }
$s = '';
foreach ($a as $v) { $s .= substr($v, 0, 3); }
$u = unserialize(serialize(array('k' => $a, 'n' => 17)));
echo 'sanity ', strlen($s), ' ', count($u['k']), ' ', $u['n'], ' ',
     implode('', $k), ' ', implode(',', $w), "\n";
