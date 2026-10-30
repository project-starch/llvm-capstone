<?php
$r = array(fopen("php://memory","r"));
$a = array(1,2,3);
@stream_select($r, $w, $e, $a, 0);
echo "done\n";
