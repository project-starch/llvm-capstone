<?php
// Bug #49785: insufficient input string validation of htmlspecialchars()/htmlentities().
// In php-5.0.0 get_next_char() (ext/standard/html.c) has NO string-length parameter
// and, for the multibyte charsets, peeks ahead with str[pos] / str[pos+1] without any
// bound check. A trailing EUC-JP JIS-X-0212 lead byte 0x8f at the very end of the buffer
// makes it read str[pos+1] one past the allocated zval string -> heap over-read.
$s = "\x8f";
echo htmlspecialchars($s, ENT_QUOTES, "EUC-JP");
