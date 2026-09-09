#define R27_OLDER sd t6, 0(s11)
#define R27_FENCE fence
#define R27_PAD .rept 80; nop; .endr
#include "r27-fence-body.h"
