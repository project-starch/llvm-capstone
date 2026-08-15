/* jmp_buf for capstone64: 224 bytes, 16-byte aligned.
 *
 * WHY IT DIFFERS FROM riscv64. Upstream declares `unsigned long __jmp_buf[26]`
 * -- 208 bytes, 8-byte aligned -- which is right for a machine whose callee-saved
 * registers are 8 bytes. Here sp and s0-s11 are 128-bit CAPABILITIES, so
 * capstone_setjmp.S stores them with `stc` at 16-byte strides:
 *
 *     sd  ra,  0(a0)          ra is a scalar under this ABI, low 8 B of its slot
 *     stc sp,  16(a0)
 *     stc s0,  32(a0)  ...  stc s11, 208(a0)      <- writes bytes 208..223
 *
 * That is 224 bytes into a 208-byte object: a 16-byte overrun of whatever
 * follows the buffer. And every `stc` needs a 16-byte-aligned address, which an
 * `unsigned long[26]` does not guarantee -- on a misaligned address stc drops
 * the tag, so a longjmp would restore an untagged sp.
 *
 * FOUND by mruby faulting in setjmp during mrb_open. The Lua probe has the same
 * mismatch and does NOT fault, which is the worse case: it silently wrote 16
 * bytes past its jmp_buf on every protected call. Anything measured on a
 * musl-linked domain that uses setjmp predates this file.
 *
 * 28 * 8 = 224. Keep this in step with capstone_setjmp.S: the last store there
 * is the one that sizes this type.
 */
typedef unsigned long __jmp_buf[28] __attribute__((__aligned__(16)));
