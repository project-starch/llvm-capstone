/* The traced zend_alloc.c's one include (zend-alloc-layout-trace.patch): the layout trace's
 * counters are defined in that file, and zend-cap-mode.c only declares them. Inert without
 * ZEND_CAP_LAYOUT_TRACE. See zend-layout-trace.h. */
#ifdef ZEND_CAP_LAYOUT_TRACE
# define ZEND_LAYOUT_TRACE_DEFINE 1
# include "zend-layout-trace.h"
#endif
