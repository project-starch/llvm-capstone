<?php
class exit_filter extends php_user_filter {
    function filter($in, $out, &$consumed, $closing) {
        exit("boom\n");
    }
}
stream_filter_register("exitf", "exit_filter");
$fp = tmpfile();
fwrite($fp, str_repeat("A", 100000));
stream_filter_append($fp, "exitf", STREAM_FILTER_READ);
rewind($fp);
echo fread($fp, 8192);
