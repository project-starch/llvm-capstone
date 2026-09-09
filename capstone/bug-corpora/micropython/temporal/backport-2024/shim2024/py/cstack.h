/* py/cstack.h did not exist in the 2024 tree; the C-stack API was introduced later.
   Our port calls exactly one function from it, so map that onto the 2024 equivalent
   in py/stackctrl.h, which the port already includes. Same two actions: record the
   current stack top, then set the limit. */
#ifndef MICROPY_INCLUDED_PY_CSTACK_H
#define MICROPY_INCLUDED_PY_CSTACK_H
#include "py/stackctrl.h"
#define mp_cstack_init_with_sp_here(n) \
    do { mp_stack_ctrl_init(); mp_stack_set_limit((mp_uint_t)(n)); } while (0)
#endif
