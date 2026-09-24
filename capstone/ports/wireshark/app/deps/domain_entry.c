/* An ordinary main() inside a musl domain, for the tshark port's third-party libraries.
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls capstone_main(); a C
 * program has main(). configure links every conftest through this adapter, so HAVE_<func> means
 * "resolves in a real domain image"; a library's own test programs link through it too. A domain
 * has no command line and no environment of its own: one argv entry, an empty environment
 * (musl's __environ is otherwise NULL, which getenv() would dereference). Same shape as
 * CPython's interpreter/toolchain/domain_entry.c and libc-test's libc_test_domain.c.
 */
extern char **__environ;
int main(int argc, char **argv);

int capstone_main(void)
{
	static char *env[] = { 0 };
	static char prog[] = "domain";
	static char *argv[] = { prog, 0 };
	__environ = env;
	return main(1, argv);
}
