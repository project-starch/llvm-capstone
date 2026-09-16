/* __set_thread_area for a pure-capability domain.
 *
 * Upstream's src/thread/riscv64/__set_thread_area.s is
 *
 *     __set_thread_area: mv tp, a0 ; li a0, 0 ; ret
 *
 * and `mv` copies 64 bits of address without the tag, so the thread pointer
 * musl installs would be unusable the moment anything dereferenced it. This is
 * the same file with the one instruction that has to change.
 *
 * It lives here rather than in arch-capstone64/ because the overlay covers
 * arch/, not src/. The archive built by build-musl-capstone.sh contains only
 * surveyed .c files, so upstream's .s is not in it and there is no duplicate
 * symbol to resolve: this definition is the only one the link sees.
 *
 * The return value is 0, which __init_tp reads as "the kernel did it", the same
 * answer upstream gives on riscv64. It is honest here for a different reason:
 * there is no kernel in the loop at all, the register is simply ours to set.
 */
.text
.global __set_thread_area
.type   __set_thread_area, %function
__set_thread_area:
	movc tp, a0
	li a0, 0
	ret
.size __set_thread_area, .-__set_thread_area
