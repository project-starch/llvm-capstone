/* A plain entry: main's status is returned. No __capstone_at_exit here, on
   purpose -- whether one exists is what the two builds of the test vary. */
extern char **__environ;
int main(int argc, char **argv);
int capstone_main(void)
{
	static char *argv[] = { "exit-test", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	return main(1, argv);
}
