#define R27_OLDER sd t6, 0(s11); .rept 80; nop; .endr
#define R27_FENCE fence
#define R27_PAD
#include "r27-fence-body.h"
