<?php
# GIT-2a31dbbadfb9  (engine-core, Zend/zend_operators.c)
# batch commit: 2a31dbbadfb90317737e9b0d4d606e75b26e0ebd "MFH:- Removed unnecessary TSRMLS_FETCH() calls"
#   -- that commit is itself cosmetic (it edits the body of convert_to_object()/convert_scalar_to_array());
#      the fault below is a DEFECT CO-LOCATED IN THE SAME FUNCTION at php-5.0.0.
#
# DEFECT (php-5.0.0 ground truth):
#   Zend/zend_variables.c:146  zval_copy_ctor(), IS_ARRAY case:
#        if (zvalue->value.ht == &EG(symbol_table)) { return; }   <-- copy DELIBERATELY SKIPPED
#      so a "copy" of $GLOBALS still ALIASES the live symbol table (a non-heap HashTable that
#      lives inside the executor_globals struct).
#   Zend/zend_operators.c:661  convert_to_object(), IS_ARRAY case:
#        object_and_properties_init(op, zend_standard_class_def, op->value.ht);
#      adopts that aliased HashTable as the new object's OWNED property table, with no
#      `op->value.ht == &EG(symbol_table)` guard.
#   Zend/zend_objects.c:88-89  zend_objects_free_object_storage():
#        zend_hash_destroy(object->properties);
#        FREE_HASHTABLE(object->properties);     -> efree(&EG(symbol_table))
#
# c_function: convert_to_object   c_file_line: Zend/zend_operators.c:661
# asan_kind: invalid-free   frame: _efree Zend/zend_alloc.c:287
#
# Effect: unset($o) tears down the LIVE global symbol table (every variable's zval is
#   destroyed) and then calls efree() on an address that was never malloc()-ed.
#
# NEIGHBOUR (disclosed, believed DISTINCT): corpus V5C-012
#   "globals-hashtable-destroy-efree-no-symtable-guard" covers ext/standard/array.c
#   array_splice()/array_unshift() (destroy+efree+replace idiom, ext locus). This one is a
#   different defective construct in a different subsystem: the Zend cast operator handing a
#   non-owned HashTable to the OBJECT STORE, sink zend_objects.c:89, and it destroys the symbol
#   table mid-request rather than replacing it.
$a = 1; $b = 2;
$o = (object)$GLOBALS;
unset($o);
$c = $a + $b;
echo "still here $c\n";
