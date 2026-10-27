	/* DIAGNOSTIC ONLY, spliced into a COPY of Zend/zend_alloc.c by build-engine.sh when
	   PHP_DIAG_ALLOC=1. csdebugprint (.insn r 0x5b, funct3 0x1, funct7 0x43) prints
	   "Print = Cap(type,perms,cursor,base,end)" when the register is TAGGED and
	   "Print = Scalar(0x..)" when it is not.
	   Three prints per _estrndup call, in this order:
	     1. p  -- what _emalloc just returned (the DESTINATION)
	     2. s  -- the SOURCE string, which is the operand that actually faults
	     3. ra -- who called _estrndup, so an untagged s names its supplier
	   The experiment arms never build this file. */
	{ void *__cap_diag_ra = __builtin_return_address(0);
	  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, zero, %0, zero" : : "r"(p));
	  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, zero, %0, zero" : : "r"(s));
	  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, zero, %0, zero" : : "r"(__cap_diag_ra)); }
