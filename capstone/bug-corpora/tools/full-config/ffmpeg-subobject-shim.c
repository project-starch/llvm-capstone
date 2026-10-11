/* FULL CONFIGURATION, FFmpeg sub-object corpus only. That corpus's driver brings up the buffer-pool
 * port's REPLAY backend (an arena for av_malloc, a payload arena, a mode, a trace) before it makes
 * its pools. In the full configuration the pools are FFmpeg's own on their Sublet port and av_malloc
 * is libavutil's on the Sublet heap, so that backend is not linked; these are the four calls the
 * driver makes into it, as no-ops. The arenas the driver still allocates go unused.
 *
 * Linking the backend instead (as the `sublet` arm does) cannot work here: its metadata-allocator.c
 * replaces av_malloc with an arena the driver initialises only in main(), so the full-config
 * constructor's pool, made before main(), got no memory (the first run's FULLCONFIG-FAILED line). */
#include <stddef.h>

void ff2_memory_init(void *base, size_t bytes) { (void)base; (void)bytes; }
void ff2_payload_init(void *p, size_t n) { (void)p; (void)n; }
void ff2_set_mode(unsigned value) { (void)value; }
void ff2_reset(void) {}
