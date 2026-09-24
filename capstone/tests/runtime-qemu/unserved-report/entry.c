/* A plain entry: main's status is returned. */
extern char **__environ;
int main(int argc, char **argv);
int capstone_main(void)
{
	static char *argv[] = { "unserved-report", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	return main(1, argv);
}
