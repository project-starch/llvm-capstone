/* The link's undefined-symbol control (link-cpython-capstone.py): linked into
 * a second copy of the interpreter link, it must make that link report exactly
 * one more undefined symbol, capstone_link_control_nowhere. A report that does
 * not see it is not reading the linker's output, and its "0 undefined" for the
 * interpreter would mean nothing. */
void capstone_link_control_nowhere(void);

void capstone_link_control_caller(void)
{
	capstone_link_control_nowhere();
}
