/* A plain entry: main's status is returned. The at-exit hook is defined here so
   no image in this test carries an undefined weak symbol (C-56). */
extern char **__environ;
int main(int argc, char **argv);
int __capstone_at_exit(int status) { return status; }
int capstone_main(void)
{
	static char *argv[] = { "tls-test", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	return main(1, argv);
}
