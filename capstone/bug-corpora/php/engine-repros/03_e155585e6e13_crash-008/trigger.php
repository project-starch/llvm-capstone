<?php
// GIT-e155585e6e13 — date("U", <big>) heap-buffer-overflow (php_date size under-estimate)
// fix sha:  e155585e6e1 "Reimplemented date and gmdate with new timelib code.
//           Removed old date/gmdate implementations." (2005, Derick) — deletes the
//           hand-rolled php_date() that carried this buffer-sizing bug.
// C site:   ext/standard/datetime.c:321  php_date()  (php-4.0.2)
//           strcat(return_value->value.str.val, tmp_buff);   // 'U' write
// cause:    The two-pass php_date() first computes an output size, then writes.
//           For format char 'U' (seconds since epoch) the size pass adds a FIXED
//           `size += 10` (datetime.c:254), but the write pass does
//           sprintf(tmp_buff,"%ld",(long)the_time) of a 64-bit time_t, which can
//           be 11-20 decimal digits.  emalloc(size+1) is therefore too small and
//           the strcat() writes past the heap buffer -> heap-buffer-overflow WRITE.
// verifier: asan_report=heap-buffer-overflow @ php_date datetime.c:321; both ports safe.
echo date("U", 999999999999999);

