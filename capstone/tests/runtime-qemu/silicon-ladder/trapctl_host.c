/* Native oracle for the trap-handler control -- and it is a SENTINEL, not a computation.
 *
 * Every other rung's host mirrors the domain's arithmetic, so the oracle is an independent
 * calculation of the same answer. There is nothing to mirror here: trapctl's verdict is a fixed
 * sentinel that the domain writes to res[0], and the question the rung asks -- does a capability
 * fault inside a domain become a RETURN rather than a wedge? -- cannot be posed to a native build at
 * all. So this oracle carries the documented PASS value and nothing else.
 *
 * WHAT THAT DOES AND DOES NOT PROVE. It does not verify the value; it fixes what the run is compared
 * against, so a QEMU run has a stated expectation rather than being read after the fact. The content
 * of the test is entirely in whether the emulated run independently REACHES 0x7A05, which requires
 * the deliberate out-of-bounds ldc to fault AND the installed handler to convert that into a return.
 * 0x7A06 -- the ldc did not fault -- is the reading this guards against, and it is why a bare
 * "did it return?" check would not do: a gate that cannot fire is not a passing gate.
 *
 * Keep this in step with trapctl_kernel.h's verdict table if the sentinels are ever renumbered. */
#include <stdio.h>

int main(void)
{
  printf("%lu\n", 0x7A05UL);   /* THE PASS: the deliberate fault happened and the handler returned */
  return 0;
}
