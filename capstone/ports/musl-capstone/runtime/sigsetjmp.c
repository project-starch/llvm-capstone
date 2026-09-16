/* sigsetjmp and siglongjmp for a domain, where there is no signal mask.
 *
 * musl's sigsetjmp saves the registers and then calls __sigsetjmp_tail, which
 * issues rt_sigprocmask to save or restore the mask. A domain has no signals
 * and no mask: the syscall would be recorded as unserved and the test would be
 * marked as having needed something it could not get. Treating the mask as
 * empty is not an approximation here, it is the state of the environment, so
 * these are setjmp and longjmp under their POSIX names. The `savemask`
 * argument is accepted and has nothing to save.
 */
#include <setjmp.h>

int sigsetjmp(sigjmp_buf env, int savemask)
{
	(void)savemask;
	return setjmp(env);
}

_Noreturn void siglongjmp(sigjmp_buf env, int val)
{
	longjmp(env, val);
}
