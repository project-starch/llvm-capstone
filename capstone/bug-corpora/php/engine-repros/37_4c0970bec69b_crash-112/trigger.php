<?php
class myfilter extends php_user_filter {
    function filter($in, $out, &$consumed, $closing) {
        while ($b = stream_bucket_make_writeable($in)) {
            stream_bucket_append($out, $b);
        }
        return PSFS_PASS_ON;
    }
}
stream_filter_register("myfilter", "myfilter");
$fn = tempnam("/tmp", "flt");
$fp = fopen($fn, "w+");
stream_filter_append($fp, "myfilter");
fwrite($fp, "test data");
fclose($fp);
echo "done\n";
