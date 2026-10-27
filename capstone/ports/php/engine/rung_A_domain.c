/* Phase 2 rung A: does the engine LINK and do its globals come up?
 * No zend_startup yet -- this rung proves the 34 TUs link against the support layer and
 * that capability-global init survives an image this size. */
extern void php_capstone_set_sink(void);
void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    *res = 0xA0u;   /* reached = the image loaded and entry glue ran */
}
