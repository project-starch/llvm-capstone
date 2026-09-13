/* What the pool needs from the level below, and nothing else. Its own header so that the patch
   to upstream ngx_palloc.c is one include line rather than an include plus three declarations
   that would have to be kept in step with ngx_subpool.c by hand. */
#ifndef NGX_SUBPOOL_H
#define NGX_SUBPOOL_H

#include "sublet.h"

void ngx_subpool_init(sublet_cap *region);
int  ngx_subpool_block(size_t bytes, sublet_cap *region, sublet_cap *handle);
void ngx_subpool_release(sublet_cap *region, sublet_cap *handle);

extern unsigned long ngx_subpool_carved, ngx_subpool_reused, ngx_subpool_returned,
                     ngx_subpool_live;

#endif
