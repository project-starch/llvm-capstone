<?php
ini_set('default_socket_timeout', 1);
$srv = stream_socket_server('tcp://127.0.0.1:12401', $e, $s);
$ctx = stream_context_create(array('http'=>array('request_fulluri'=>true)));
@fopen('http://127.0.0.1:12401/', 'r', false, $ctx);
echo "done\n";
