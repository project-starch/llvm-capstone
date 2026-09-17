/* Reduced from av_buffer_ref: the main installed compiler fails selecting
 * this atomic add through its default AS200 pointer at -O0 with +a enabled.
 * Define FFPOOL_PLAIN_COUNTER for the ordinary-load/store control.
 * This is a reproduction input, not an attempted compiler fix.
 */
#ifdef FFPOOL_PLAIN_COUNTER
unsigned retain(unsigned *count) { return (*count)++; }
#else
unsigned retain(_Atomic(unsigned) *count)
{
    return __c11_atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
}
#endif
