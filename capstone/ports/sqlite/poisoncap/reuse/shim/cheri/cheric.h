/* Capstone stand-in for CheriBSD's <cheri/cheric.h>: only what their speedtest1.c uses. cheri_gettag
 * answers 1: their start-up lines "ptr is tagged.." then describe this shim, not a tag. */
#pragma once
#define cheri_gettag(p) 1
