/* WAMR calls bh_assert, which routes here. NDEBUG builds drop it entirely. */
#ifndef CAPSTONE_WAMR_ASSERT_H
#define CAPSTONE_WAMR_ASSERT_H
#ifdef NDEBUG
#define assert(e) ((void)0)
#else
void abort(void);
#define assert(e) ((e) ? (void)0 : abort())
#endif

/* C11 spells it _Static_assert; WAMR writes static_assert and expects <assert.h>
   to provide the alias, which a hosted libc does. */
#ifndef static_assert
#define static_assert _Static_assert
#endif
#endif
