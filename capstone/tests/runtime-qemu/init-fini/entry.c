/* A plain entry: main's status is returned. The environment comes from
 * __capstone_domain_environ, which the runtime calls before the constructors
 * run (C-64), so a constructor can read it, as in C. */
int main(int argc, char **argv);
char **__capstone_domain_environ(void)
{
	static char *envp[] = { "INIT_FINI_ENV=before-main", 0 };
	return envp;
}
int capstone_main(void)
{
	static char *argv[] = { "init-fini", 0 };
	return main(1, argv);
}
