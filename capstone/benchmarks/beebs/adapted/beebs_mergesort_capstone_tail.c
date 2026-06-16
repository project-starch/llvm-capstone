/* Capstone-adapted tail for the BEEBS mergesort benchmark.
 *
 * Replaces everything from `typedef bool (*Comparison)` onwards with
 * function-pointer-free equivalents.  Function pointers in Capstone domains
 * require capability-tagged values.  The domain ELF loader does not process
 * relocations, so function pointer arrays in .data are untagged and `cjalr`
 * through them faults.  This tail hard-codes the comparison inline.
 *
 * Also provides:
 *  - memcpy stub (not in freestanding libc, used by MergeSortR)
 *  - alloca replacement: static buffer inside MergeSort
 *  - integer replacements for the FP jitter test-data functions
 *  - benchmark() using a switch rather than a function pointer dispatch table
 *  - verify_benchmark() with global const arrays (avoids stc bulk-copy Bug #9)
 *
 * Expected values come from a host reference run with the same integer
 * formulas.  Run the host capture program to regenerate if the jitter
 * formulas change.
 *
 * Bug #10 — Range by-value struct ABI (stc upper-half zeroing):
 *  On Capstone, a 16-byte struct like Range is passed/returned as a single
 *  128-bit capability slot.  The compiler loads only Range.start (64 bits)
 *  via `ld`, then stores the 128-bit slot via `stc`, which zeroes the upper
 *  half — clobbering Range.end.  Fix: all sort functions take `const Range *`
 *  instead of `const Range`; struct fields are assigned individually so the
 *  compiler emits separate `sd` instructions rather than a struct copy.
 *  MakeRange and Range_length from the upstream prefix are intentionally
 *  avoided in this file.
 */

/* memcpy stub — satisfies MergeSortR's calls without hosted <string.h> */
void *memcpy(void *dest, const void *src, unsigned long n)
{
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    unsigned long i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
    return dest;
}

/* --- Sort functions: Range by pointer, comparison inlined, no fn-ptr --- */

long BinaryLast(const Test array[], const long index, const Range *range)
{
    long start = range->start, end = range->end - 1;
    while (start < end) {
        long mid = start + (end - start) / 2;
        if (!(array[index].value < array[mid].value))
            start = mid + 1;
        else
            end = mid;
    }
    if (start == range->end - 1 && !(array[index].value < array[start].value))
        start++;
    return start;
}

void InsertionSort(Test array[], const Range *range)
{
    long i;
    for (i = range->start + 1; i < range->end; i++) {
        const Test temp = array[i];
        long j;
        for (j = i; j > range->start && (temp.value < array[j - 1].value); j--)
            array[j] = array[j - 1];
        array[j] = temp;
    }
}

void MergeSortR(Test array[], const Range *range, Test buffer[])
{
    long mid, A_count = 0, B_count = 0, insert = 0;
    long A_len, B_len;
    Range A, B;

    if (range->end - range->start < 32) {
        InsertionSort(array, range);
        return;
    }

    mid = range->start + (range->end - range->start) / 2;
    A.start = range->start;
    A.end   = mid;
    B.start = mid;
    B.end   = range->end;

    MergeSortR(array, &A, buffer);
    MergeSortR(array, &B, buffer);

    A.start = BinaryLast(array, B.start, &A);
    A_len = A.end - A.start;
    B_len = B.end - B.start;

    memcpy(&buffer[0], &array[A.start], A_len * sizeof(array[0]));
    while (A_count < A_len && B_count < B_len) {
        if (!(array[A.end + B_count].value < buffer[A_count].value)) {
            array[A.start + insert] = buffer[A_count];
            A_count++;
        } else {
            array[A.start + insert] = array[A.end + B_count];
            B_count++;
        }
        insert++;
    }
    memcpy(&array[A.start + insert], &buffer[A_count],
           (A_len - A_count) * sizeof(array[0]));
}

void MergeSort(Test array[], const long array_count)
{
    static Test _buffer_local[100];
    Range r;
    r.start = 0;
    r.end   = array_count;
    MergeSortR(array, &r, _buffer_local);
}

/* --- Test-data generators (integer replacements for FP jitter) --- */

long TestingPathological(long index, long total)
{
    if (index == 0) return 10;
    else if (index < total / 2) return 11;
    else if (index == total - 1) return 10;
    return 9;
}

long TestingRandom(long index, long total)
{
    (void)index; (void)total;
    return rand_beebs();
}

/* Integer approximation: rand * 5/RAND_MAX - 2 replaces rand*1.0/RAND_MAX*5-2.5 */
long TestingMostlyDescending(long index, long total)
{
    return total - index + rand_beebs() * 5 / RAND_MAX - 2;
}

long TestingMostlyAscending(long index, long total)
{
    return index + rand_beebs() * 5 / RAND_MAX - 2;
}

long TestingAscending(long index, long total)
{
    (void)total;
    return index;
}

long TestingDescending(long index, long total)
{
    return total - index;
}

long TestingEqual(long index, long total)
{
    (void)index; (void)total;
    return 1000;
}

/* Integer approximation: compare rand against RAND_MAX*9/10 */
long TestingJittered(long index, long total)
{
    (void)total;
    return (rand_beebs() <= RAND_MAX * 9 / 10) ? index : (index - 2);
}

/* Integer approximation: rand*4/RAND_MAX replaces rand*1.0/RAND_MAX*4 */
long TestingMostlyEqual(long index, long total)
{
    (void)index; (void)total;
    return 1000 + rand_beebs() * 4 / RAND_MAX;
}

/* --- Global state --- */

const long max_size = 100;
Test array1[100];

void initialise_benchmark(void) {}

/* --- benchmark: switch dispatch instead of function pointer table --- */

int benchmark(void)
{
    long total, index, test_case;

    total = max_size;
    for (test_case = 0; test_case < 9; test_case++) {
        for (index = 0; index < total; index++) {
            long v;
            Test item;

            switch (test_case) {
            case 0: v = TestingPathological(index, total); break;
            case 1: v = TestingRandom(index, total);       break;
            case 2: v = TestingMostlyDescending(index, total); break;
            case 3: v = TestingMostlyAscending(index, total);  break;
            case 4: v = TestingAscending(index, total);    break;
            case 5: v = TestingDescending(index, total);   break;
            case 6: v = TestingEqual(index, total);        break;
            case 7: v = TestingJittered(index, total);     break;
            default: v = TestingMostlyEqual(index, total); break;
            }

            item.value = (int)v;
            item.index = (int)index;
            array1[index] = item;
        }
        MergeSort(array1, total);
    }
    return 0;
}

/* --- verify_benchmark: global arrays avoid stc bulk-copy Bug #9 --- */

static const int exp_val[100] = {
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1001, 1001,
    1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001,
    1001, 1001, 1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002,
    1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002,
    1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 1002,
    1002, 1002, 1002, 1003, 1003, 1003, 1003, 1003, 1003, 1003,
    1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003,
    1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003
};

static const int exp_index[100] = {
     9, 11, 19, 21, 29, 31, 33, 34, 36, 39,
    40, 42, 46, 47, 49, 52, 54, 59, 63, 67,
    69, 73, 76, 79, 84, 86, 93, 94,  7, 12,
    13, 15, 16, 23, 48, 64, 65, 68, 70, 71,
    72, 77, 80, 83, 97, 98,  2,  4,  5,  6,
    10, 14, 17, 22, 27, 28, 32, 37, 53, 55,
    57, 61, 62, 66, 74, 75, 81, 87, 89, 91,
    92, 95, 99,  0,  1,  3,  8, 18, 20, 24,
    25, 26, 30, 35, 38, 41, 43, 44, 45, 50,
    51, 56, 58, 60, 78, 82, 85, 88, 90, 96
};

int verify_benchmark(int unused)
{
    int i;
    for (i = 0; i < (int)max_size; i++)
        if (array1[i].value != exp_val[i])
            return 0;
    for (i = 0; i < (int)max_size; i++)
        if (array1[i].index != exp_index[i])
            return 0;
    return 1;
}
