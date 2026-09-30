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
.size __capstone_call_on_stack, .-__capstone_call_on_stack
