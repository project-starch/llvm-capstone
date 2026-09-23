/* An ordinary main() inside a musl domain.
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls
 * capstone_main(); a C program has main(). This is the adapter, and the same
 * shape as libc-test's libc_test_domain.c without the test harness: an empty
 * environment rather than musl's null __environ (which __libc_start_main would
 * have set, and a domain does not run), and one argv entry.
 *
 * configure links every conftest through it, so HAVE_<func> means "resolves in
 * a real domain image", and CPython's own Programs/python.c main() will land
 * here unchanged when the interpreter first links.
 */
extern char **__environ;
int main(int argc, char **argv);

int capstone_main(void)
{
	static char *argv[] = { "python3", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	return main(1, argv);
}
