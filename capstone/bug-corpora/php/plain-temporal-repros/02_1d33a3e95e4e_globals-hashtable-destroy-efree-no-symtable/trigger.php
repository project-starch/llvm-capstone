<?php
# GIT-1d33a3e95e4e  (distinct site: ext/standard/array.c:1645)
# fix sha: 1d33a3e95e4e23188ca7fb4d289a96d1fa4be50e (2005-07-04)
# fix subject: "Fixed bug #31158 (array_splice on $GLOBALS crashes)"
#   -> the fix adds `if (Z_ARRVAL_P(array) == &EG(symbol_table)) zend_reset_all_cv(...)`
#      and copies the new hashtable in place instead of efree()ing the old one.
# C file:line: ext/standard/array.c:1645  (ASan surface frame: _efree Zend/zend_alloc.c:223)
#   1644  zend_hash_destroy(array->value.ht);
#   1645  efree(array->value.ht);            <-- frees EG(symbol_table)'s HashTable
#   1646  array->value.ht = new_hash;
# c_function: php_if_array_splice
# cause: array_splice() destroys+efree()s the argument array's HashTable and swaps in a
#   fresh one. When the argument is $GLOBALS, that HashTable IS the interpreter's active
#   symbol table (not a separately malloc()-ed block), so efree() faults with
#   "attempting free on address which was not malloc()-ed" (invalid-free). Present at
#   php-4.0.2 (array.c v: PHP_FUNCTION(array_splice) uses the destroy+efree+replace idiom
#   with no symbol-table guard). DISTINCT from catalogued array.c:525/617/989 (sort/walk
#   callback UAFs) and from zend_alloc.c:239 (unserialize size bomb): different line,
#   different mechanism (symbol-table invalid-free).
# port_implements: true (port runs array_splice on $GLOBALS and prints "done" -- both
#   memory models keep the symbol table intact; no invalid free).
$a=1;$b=2;$c=3;
array_splice($GLOBALS, 0, 1);
echo "done\n";
