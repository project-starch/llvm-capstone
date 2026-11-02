<?php
class strtoupper_filter extends php_user_filter {
    function filter($in, $out, &$consumed, $closing) {
        while ($bucket = stream_bucket_make_writeable($in)) {
            $bucket->data = strtoupper($bucket->data);
            $consumed += $bucket->datalen;
            stream_bucket_append($out, $bucket);
            stream_bucket_append($out, $bucket);
        }
        return PSFS_PASS_ON;
    }
}
stream_filter_register("strtoupper", "strtoupper_filter");
$file = "/tmp/bug35916_test.txt";
$fp = fopen($file, "w");
stream_filter_append($fp, "strtoupper");
fwrite($fp, "Thank you\n");
fclose($fp);
readfile($file);
unlink($file);
