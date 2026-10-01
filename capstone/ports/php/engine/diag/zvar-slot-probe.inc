	/* DIAGNOSTIC ONLY (PHP_DIAG_ZVAR=1), spliced into a COPY of Zend/zend_variables.c before
	   _zval_copy_ctor's estrndup. Prints the ADDRESS of the slot that holds the string pointer,
	   so CAPSTONE_TAGWATCH_GRANULE can be aimed at the one 16-byte granule that matters. The
	   address is taken by & on zvalue, so it is capability-derived and prints as a Cap whose
	   cursor is the slot address. */
	{ void *__cap_slot = (void *)&zvalue->value.str.val;
	  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, zero, %0, zero" : : "r"(__cap_slot)); }
