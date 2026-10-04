/* void __capstone_call_on_stack(void (*fn)(void *), void *arg, void *top);
 *
 * Call fn(arg) with the stack pointer moved to `top`, the alternate signal
 * stack's upper end, and move it back afterwards. `top` is a capability whose
 * bounds are the stack the application registered with sigaltstack, so a
 * handler that overruns it faults at the boundary instead of writing into
 * whatever lies below. Pure-capability code: cjalr through the function
 * capability, as the compiler calls a function pointer. */
.text
.global __capstone_call_on_stack
.type __capstone_call_on_stack, %function
__capstone_call_on_stack:
#ifdef CAPSTONE_GP_CAPTABLE_ABI
	/* The silicon ABI (B0, docs/plans/b0-silicon-delegated-runtime.md): a function pointer and the return
	   address are plain integers inside the domain's code capability -- compiled code calls with jalr, saves
	   ra with sd and returns with ret (b0-hello.dom: 46 `sd ra`, 0 compiled cjalr). A cjalr through such an
	   integer faults. sp, s0 and arg are still capabilities. */
	cincoffsetimm sp, sp, -32
	sd ra, 16(sp)
	stc s0, 0(sp)
	movc s0, sp
	movc sp, a2
	mv t0, a0
	movc a0, a1
	jalr ra, 0(t0)
	movc sp, s0
	ld ra, 16(sp)
	ldc s0, 0(sp)
	cincoffsetimm sp, sp, 32
	ret
#else
	cincoffsetimm sp, sp, -32
	stc ra, 16(sp)
	stc s0, 0(sp)
	movc s0, sp
	movc sp, a2
	movc t0, a0
	movc a0, a1
	cjalr ra, 0(t0)
	movc sp, s0
	ldc ra, 16(sp)
	ldc s0, 0(sp)
	cincoffsetimm sp, sp, 32
	cjalr zero, 0(ra)
#endif
.size __capstone_call_on_stack, .-__capstone_call_on_stack
