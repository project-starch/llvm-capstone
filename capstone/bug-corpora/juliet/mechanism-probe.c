/* The sizes that decide Juliet's CWE-122 `sizeof_double/int64_t/struct` family: the case allocates
 * malloc(sizeof(data)) with data a pointer, then stores ONE element. On this ABI a pointer is a 16-byte capability, so
 * the allocation holds the element whenever sizeof(T*) >= sizeof(T). Built with the virtual SDK and run in the guest by
 * probe.py, so the sizes are the ones the cases were compiled with. */
#include <stdio.h>
#include <stdint.h>

typedef struct { int intOne; int intTwo; } twoIntsStruct;   /* Juliet's std_testcase.h */

int main(void) {
  printf("PROBE sizeof-ptr double*=%zu double=%zu int64_t*=%zu int64_t=%zu struct*=%zu struct=%zu\n",
         sizeof(double *), sizeof(double), sizeof(int64_t *), sizeof(int64_t),
         sizeof(twoIntsStruct *), sizeof(twoIntsStruct));
  return 0;
}
