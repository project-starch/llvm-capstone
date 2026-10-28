<?php
/* GIT-fd5bd37ab129 — Revert "Fixed bug #75961 (Strange references behavior)".
 *
 * The reverted PHP-7 commit made php_array_walk() access the ARRAY SLOT rather
 * than a stable reference; if the userland callback modifies the array being
 * walked, the slot is freed underneath the loop -> use-after-free. The fault
 * CLASS is "array_walk callback mutates the walked array".
 *
 * At php-4.0.2 the same class is independently live. php_array_walk()
 * (php-4.0.2:ext/standard/array.c:948) iterates the hash with the internal
 * pointer, handing each bucket's data to call_user_function_ex (array.c:980)
 * and then reading the current key (array.c:989 ->
 * zend_hash_get_current_key_type_ex, zend_hash.c:1020) and moving forward.
 * Nothing protects that internal pointer / bucket against the callback: when
 * the callback destroys the very array being walked, the buckets are efree'd by
 * _zval_dtor, and the next zend_hash_get_current_key_type_ex dereferences freed
 * memory -> heap-use-after-free.
 *
 * Trigger: walk a global array whose callback reassigns/unsets that same global
 * array, freeing its buckets mid-iteration. */

$GLOBALS['arr'] = array();
for ($i = 0; $i < 200; $i++) {
    $GLOBALS['arr']["k$i"] = "value_string_$i";
}

function cb(&$v, $k) {
    /* Destroy the array currently being walked, from inside the callback.
     * This efree()s the buckets the walk loop still holds an internal
     * pointer into. */
    $GLOBALS['arr'] = 0;
    unset($GLOBALS['arr']);
    $GLOBALS['arr'] = array("x" => str_repeat("Z", 64));
}

array_walk($GLOBALS['arr'], "cb");
echo "done\n";
