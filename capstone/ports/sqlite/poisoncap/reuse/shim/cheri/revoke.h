/* Capstone stand-in for CheriBSD's <cheri/revoke.h>: their speedtest1.c calls cheri_revoke once at
 * start-up; Capstone has no sweeping revoker, so the call is a no-op. */
#pragma once
#include <stdint.h>
struct cheri_revoke_syscall_info { struct { uint64_t enqueue, dequeue; } epochs; };
#define CHERI_REVOKE_LAST_PASS 0
#define CHERI_REVOKE_IGNORE_START 0
#define CHERI_REVOKE_TAKE_STATS 0
static inline int cheri_revoke(int flags, unsigned long start, struct cheri_revoke_syscall_info *i) {
  (void)flags; (void)start; (void)i; return 0; }
