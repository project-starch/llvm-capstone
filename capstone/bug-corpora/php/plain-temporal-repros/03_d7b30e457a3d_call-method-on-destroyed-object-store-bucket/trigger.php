<?php
class wrapper {
    function stream_open() { return true; }
    function stream_eof() { throw new Exception(); }
}
stream_wrapper_register("wrap", "wrapper");
$fp = fopen("wrap://...", "r");
feof($fp);
echo "Done\n";
