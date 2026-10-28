<?php
// bug #67399: putenv() with an empty variable name (setting starts with '=')
// First call registers a putenv_entry keyed by the empty key and leaves its
// putenv_string live inside libc's environ. The second empty-key call runs
// zend_hash_del on the empty key, freeing that putenv_string while environ
// still points at it; the following environ scan reads the freed buffer.
putenv("=");
putenv("=foo");
