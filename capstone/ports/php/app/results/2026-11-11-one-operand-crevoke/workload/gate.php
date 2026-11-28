<?php
$a = array(3, 1, 2); sort($a); $f = fopen("/tmp/virtual-php", "w"); fwrite($f, "virtual"); fclose($f); $s = "PHP_VIRTUAL_OK sorted=" . implode(",", $a) . " md5=" . md5("capstone") . " file=" . file_get_contents("/tmp/virtual-php") . "\n"; echo $s;
