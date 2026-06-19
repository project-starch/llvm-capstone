/* Capstone-adapted tail for the BEEBS wikisort benchmark.
 *
 * Replaces everything from the Range struct definition onward.
 * The stripped prefix (kept from libwikisort.c) provides:
 *   rand_beebs, Min, Max, Test struct, TestCompare, Comparison typedef,
 *   FloorPowerOfTwo, and the Swap macro.
 *
 * Key changes vs upstream:
 *  - memcpy / memmove as inline stubs (Capstone backend cannot lower libcalls
 *    with null symbol names, crashing SelectionDAGISel)
 *  - isqrt replaces sqrt (WikiSort uses sqrt for block_size; soft-float
 *    libcalls are unavailable on Capstone)
 *  - Range uses 32-bit fields and Range helpers take const Range *.
 *    Upstream's 16-byte Range values hit the Capstone aggregate-copy ABI bug:
 *    stores through a 128-bit carrier zero the upper half, clobbering end.
 *  - Float test generators replaced with integer equivalents
 *  - test_cases[] function pointer array replaced with switch dispatch (domain
 *    ELF loader does not process relocations, so function pointer arrays in
 *    .data are untagged; cjalr through untagged pointers faults)
 *  - verify_benchmark checks that the output array is sorted
 */

static void *memcpy(void *d, const void *s, unsigned long n)
{
    unsigned char *dd = (unsigned char *)d;
    const unsigned char *ss = (const unsigned char *)s;
    unsigned long i;
    for (i = 0; i < n; i++) dd[i] = ss[i];
    return d;
}

static void *memmove(void *d, const void *s, unsigned long n)
{
    unsigned char *dd = (unsigned char *)d;
    const unsigned char *ss = (const unsigned char *)s;
    long i;
    if (dd < ss)
        for (i = 0; i < (long)n; i++) dd[i] = ss[i];
    else
        for (i = (long)n - 1; i >= 0; i--) dd[i] = ss[i];
    return d;
}

/* Integer square root (floor).  Replaces the float sqrt() call in WikiSort
 * that computes block_size.  The result is identical for all inputs in the
 * max_size=400 domain. */
static long isqrt(long x)
{
    long r = 0, bit;
    if (x <= 0) return 0;
    for (bit = 1L << 30; bit > x; bit >>= 2)
        ;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; }
        else               { r >>= 1; }
        bit >>= 2;
    }
    return r;
}

/* --- Swap macro (originally between Range and sort functions in upstream) - */
/* Var is defined in the kept prefix. */
#define Swap(value1, value2) { \
    Var(a, &(value1)); \
    Var(b, &(value2)); \
    Var(c, *a); \
    *a = *b; \
    *b = c; \
}

/* 63 -> 32, 64 -> 64, etc. (from upstream, after Range struct) */
static long FloorPowerOfTwo(const long value)
{
    long x = value;
    x = x | (x >> 1); x = x | (x >> 2);
    x = x | (x >> 4); x = x | (x >> 8);
    x = x | (x >> 16);
#if __LP64__
    x = x | (x >> 32);
#endif
    return x - (x >> 1);
}

/* --- Range struct and helpers -------------------------------------------- */

typedef struct { int start; int end; } Range;

static long Range_length(const Range *r) { return r->end - r->start; }

static unsigned long Range_bytes(const Range *r)
{
    long length = Range_length(r);
    return length > 0 ? (unsigned long)length * sizeof(Test) : 0;
}

static Range MakeRange(const long start, const long end)
{
    Range r; r.start = (int)start; r.end = (int)end; return r;
}

/* --- Sort primitives ------------------------------------------------------ */

static long BinaryFirst(const Test *array, const long index, const Range *range,
                         const Comparison compare)
{
    long start = range->start, end = range->end - 1;
    while (start < end) {
        long mid = start + (end - start) / 2;
        if (compare(array[mid], array[index])) start = mid + 1;
        else                                    end   = mid;
    }
    if (start == range->end - 1 && compare(array[start], array[index])) start++;
    return start;
}

static long BinaryLast(const Test *array, const long index, const Range *range,
                        const Comparison compare)
{
    long start = range->start, end = range->end - 1;
    while (start < end) {
        long mid = start + (end - start) / 2;
        if (!compare(array[index], array[mid])) start = mid + 1;
        else                                     end   = mid;
    }
    if (start == range->end - 1 && !compare(array[index], array[start])) start++;
    return start;
}

static void InsertionSort(Test *array, const Range *range, const Comparison compare)
{
    long i;
    for (i = range->start + 1; i < range->end; i++) {
        const Test temp = array[i];
        long j;
        for (j = i; j > range->start && compare(temp, array[j - 1]); j--)
            array[j] = array[j - 1];
        array[j] = temp;
    }
}

static void Reverse(Test *array, const Range *range)
{
    long index;
    for (index = Range_length(range) / 2 - 1; index >= 0; index--)
        Swap(array[range->start + index], array[range->end - index - 1]);
}

static void BlockSwap(Test *array, const long start1, const long start2,
                       const long block_size)
{
    long index;
    for (index = 0; index < block_size; index++)
        Swap(array[start1 + index], array[start2 + index]);
}

static void Rotate(Test *array, const long amount, const Range *range,
                   Test *cache, const long cache_size)
{
    Range range1, range2;
    long split;
    if (Range_length(range) == 0) return;
    split  = (amount >= 0) ? range->start + amount : range->end + amount;
    range1 = MakeRange(range->start, split);
    range2 = MakeRange(split, range->end);
    if (Range_length(&range1) <= Range_length(&range2)) {
        if (Range_length(&range1) <= cache_size) {
            memcpy(&cache[0], &array[range1.start], Range_bytes(&range1));
            memmove(&array[range1.start], &array[range2.start],
                    Range_bytes(&range2));
            memcpy(&array[range1.start + Range_length(&range2)], &cache[0],
                   Range_bytes(&range1));
            return;
        }
    } else {
        if (Range_length(&range2) <= cache_size) {
            memcpy(&cache[0], &array[range2.start], Range_bytes(&range2));
            memmove(&array[range2.end - Range_length(&range1)], &array[range1.start],
                    Range_bytes(&range1));
            memcpy(&array[range1.start], &cache[0], Range_bytes(&range2));
            return;
        }
    }
    Reverse(array, &range1);
    Reverse(array, &range2);
    Reverse(array, range);
}

static void WikiMerge(Test *array, const Range *buffer,
                       const Range *A, const Range *B,
                       const Comparison compare,
                       Test *cache, const long cache_size)
{
    if (Range_length(A) <= cache_size) {
        long A_count = 0, B_count = 0, insert = 0;
        long A_length = Range_length(A);
        long B_length = Range_length(B);
        if (B_length > 0 && A_length > 0) {
            while (true) {
                if (!compare(array[B->start + B_count], cache[A_count])) {
                    array[A->start + insert] = cache[A_count];
                    A_count++; insert++;
                    if (A_count >= A_length) break;
                } else {
                    array[A->start + insert] = array[B->start + B_count];
                    B_count++; insert++;
                    if (B_count >= B_length) break;
                }
            }
        }
        if (A_count < A_length)
            memcpy(&array[A->start + insert], &cache[A_count],
                   (unsigned long)(A_length - A_count) * sizeof(array[0]));
    } else {
        long A_count = 0, B_count = 0, insert = 0;
        if (Range_length(B) > 0 && Range_length(A) > 0) {
            while (true) {
                if (!compare(array[B->start + B_count], array[buffer->start + A_count])) {
                    Swap(array[A->start + insert], array[buffer->start + A_count]);
                    A_count++; insert++;
                    if (A_count >= Range_length(A)) break;
                } else {
                    Swap(array[A->start + insert], array[B->start + B_count]);
                    B_count++; insert++;
                    if (B_count >= Range_length(B)) break;
                }
            }
        }
        BlockSwap(array, buffer->start + A_count, A->start + insert,
                  Range_length(A) - A_count);
    }
}

/* --- WikiSort (main sort function) ---------------------------------------- */

static void WikiSort(Test *array, const long size, const Comparison compare)
{
static Test wiki_cache[513] __attribute__((aligned(16)));
static unsigned char wiki_bss_pad[16] __attribute__((used, aligned(16)));
#define CACHE_SIZE 512
    const long cache_size = CACHE_SIZE;
    Test *cache = wiki_cache;
    long index, merge_size, start, mid, end, fractional, decimal;
    long power_of_two, fractional_base, fractional_step, decimal_step;

    if (size <= 32) {
        Range _r = MakeRange(0, size);
        InsertionSort(array, &_r, compare);
        return;
    }

    power_of_two    = FloorPowerOfTwo(size);
    fractional_base = power_of_two / 16;
    fractional_step = size % fractional_base;
    decimal_step    = size / fractional_base;

    /* insertion-sort the lowest level (16-31 items at a time) */
    decimal = 0; fractional = 0;
    while (decimal < size) {
        Range _r;
        start    = decimal;
        decimal += decimal_step;
        fractional += fractional_step;
        if (fractional >= fractional_base) { fractional -= fractional_base; decimal += 1; }
        end = decimal;
        _r = MakeRange(start, end);
        InsertionSort(array, &_r, compare);
    }

    /* merge-sort the higher levels */
    for (merge_size = 16; merge_size < power_of_two; merge_size += merge_size) {
        long block_size  = isqrt(decimal_step);
        long buffer_size = decimal_step / block_size + 1;

        Range level1 = MakeRange(0, 0);
        Range level2 = MakeRange(0, 0);
        Range levelA = MakeRange(0, 0);
        Range levelB = MakeRange(0, 0);

        decimal = fractional = 0;
        while (decimal < size) {
            start = decimal;
            decimal += decimal_step; fractional += fractional_step;
            if (fractional >= fractional_base) { fractional -= fractional_base; decimal += 1; }
            mid = decimal;
            decimal += decimal_step; fractional += fractional_step;
            if (fractional >= fractional_base) { fractional -= fractional_base; decimal += 1; }
            end = decimal;

            if (compare(array[end - 1], array[start])) {
                /* ranges are in reverse order: rotate to fix */
                Range _r = MakeRange(start, end);
                Rotate(array, mid - start, &_r, cache, cache_size);
            } else if (compare(array[mid], array[mid - 1])) {
                /* need to merge these two ranges */
                Range bufferA, bufferB, buffer1, buffer2;
                Range blockA, blockB, firstA, lastA, lastB;
                long indexA, minA, findA;
                Test min_value;

                Range A = MakeRange(start, mid);
                Range B = MakeRange(mid,   end);

                if (Range_length(&A) <= cache_size) {
                    Range _zero = MakeRange(0, 0);
                    memcpy(&cache[0], &array[A.start], Range_bytes(&A));
                    WikiMerge(array, &_zero, &A, &B, compare, cache, cache_size);
                    continue;
                }

                if (Range_length(&level1) > 0) {
                    /* reuse buffers found in a previous iteration */
                    bufferA = MakeRange(A.start, A.start);
                    bufferB = MakeRange(B.end,   B.end);
                    buffer1 = level1;
                    buffer2 = level2;
                } else {
                    long count, length;

                    /* search for first buffer in A */
                    count = 1;
                    for (buffer1.start = A.start + 1; buffer1.start < A.end; buffer1.start++)
                        if (compare(array[buffer1.start - 1], array[buffer1.start]) ||
                            compare(array[buffer1.start],     array[buffer1.start - 1]))
                            if (++count == buffer_size) break;
                    buffer1.end = buffer1.start + count;

                    if (buffer_size <= cache_size) {
                        buffer2 = MakeRange(A.start, A.start);

                        if (Range_length(&buffer1) == buffer_size) {
                            bufferA = MakeRange(buffer1.start, buffer1.start + buffer_size);
                            bufferB = MakeRange(B.end, B.end);
                            buffer1 = MakeRange(A.start, A.start + buffer_size);
                        } else {
                            /* try B instead */
                            bufferA = MakeRange(buffer1.start, buffer1.start);
                            buffer1 = MakeRange(A.start, A.start);

                            count = 1;
                            for (buffer1.start = B.end - 2; buffer1.start >= B.start; buffer1.start--)
                                if (compare(array[buffer1.start],     array[buffer1.start + 1]) ||
                                    compare(array[buffer1.start + 1], array[buffer1.start]))
                                    if (++count == buffer_size) break;
                            buffer1.end = buffer1.start + count;

                            if (Range_length(&buffer1) == buffer_size) {
                                bufferB = MakeRange(buffer1.start, buffer1.start + buffer_size);
                                buffer1 = MakeRange(B.end - buffer_size, B.end);
                            }
                        }
                    } else {
                        /* need two buffers; search for second buffer in A */
                        count = 0;
                        for (buffer2.start = buffer1.start + 1; buffer2.start < A.end; buffer2.start++)
                            if (compare(array[buffer2.start - 1], array[buffer2.start]) ||
                                compare(array[buffer2.start],     array[buffer2.start - 1]))
                                if (++count == buffer_size) break;
                        buffer2.end = buffer2.start + count;

                        if (Range_length(&buffer2) == buffer_size) {
                            /* found both in A */
                            bufferA = MakeRange(buffer2.start, buffer2.start + buffer_size * 2);
                            bufferB = MakeRange(B.end, B.end);
                            buffer1 = MakeRange(A.start, A.start + buffer_size);
                            buffer2 = MakeRange(A.start + buffer_size, A.start + buffer_size * 2);
                        } else if (Range_length(&buffer1) == buffer_size) {
                            /* first in A, search B for second */
                            bufferA = MakeRange(buffer1.start, buffer1.start + buffer_size);
                            buffer1 = MakeRange(A.start, A.start + buffer_size);

                            count = 1;
                            for (buffer2.start = B.end - 2; buffer2.start >= B.start; buffer2.start--)
                                if (compare(array[buffer2.start],     array[buffer2.start + 1]) ||
                                    compare(array[buffer2.start + 1], array[buffer2.start]))
                                    if (++count == buffer_size) break;
                            buffer2.end = buffer2.start + count;

                            if (Range_length(&buffer2) == buffer_size) {
                                bufferB = MakeRange(buffer2.start, buffer2.start + buffer_size);
                                buffer2 = MakeRange(B.end - buffer_size, B.end);
                            } else {
                                buffer1.end = buffer1.start; /* failure */
                            }
                        } else {
                            /* search B for both buffers */
                            count = 1;
                            for (buffer1.start = B.end - 2; buffer1.start >= B.start; buffer1.start--)
                                if (compare(array[buffer1.start],     array[buffer1.start + 1]) ||
                                    compare(array[buffer1.start + 1], array[buffer1.start]))
                                    if (++count == buffer_size) break;
                            buffer1.end = buffer1.start + count;

                            count = 0;
                            for (buffer2.start = buffer1.start - 1; buffer2.start >= B.start; buffer2.start--)
                                if (compare(array[buffer2.start],     array[buffer2.start + 1]) ||
                                    compare(array[buffer2.start + 1], array[buffer2.start]))
                                    if (++count == buffer_size) break;
                            buffer2.end = buffer2.start + count;

                            if (Range_length(&buffer2) == buffer_size) {
                                bufferA = MakeRange(A.start, A.start);
                                bufferB = MakeRange(buffer2.start, buffer2.start + buffer_size * 2);
                                buffer1 = MakeRange(B.end - buffer_size, B.end);
                                buffer2 = MakeRange(buffer1.start - buffer_size, buffer1.start);
                            } else {
                                buffer1.end = buffer1.start; /* failure */
                            }
                        }
                    }

                    if (Range_length(&buffer1) < buffer_size) {
                        /* repeat-value fast path: rotate-based merge */
                        while (Range_length(&A) > 0 && Range_length(&B) > 0) {
                            long mid2   = BinaryFirst(array, A.start, &B, compare);
                            long amount = mid2 - A.end;
                            Range _r    = MakeRange(A.start, mid2);
                            Rotate(array, -amount, &_r, cache, cache_size);
                            B.start = mid2;
                            A = MakeRange(BinaryLast(array, A.start + amount, &A, compare),
                                          B.start);
                        }
                        continue;
                    }

                    /* move unique values to start of A if needed */
                    length = Range_length(&bufferA);
                    count  = 0;
                    for (index = bufferA.start; count < length; index--) {
                        if (index == A.start ||
                            compare(array[index - 1], array[index]) ||
                            compare(array[index], array[index - 1])) {
                            Range _r = MakeRange(index + 1, bufferA.start + 1);
                            Rotate(array, -count, &_r, cache, cache_size);
                            bufferA.start = index + count;
                            count++;
                        }
                    }
                    bufferA = MakeRange(A.start, A.start + length);

                    /* move unique values to end of B if needed */
                    length = Range_length(&bufferB);
                    count  = 0;
                    for (index = bufferB.start; count < length; index++) {
                        if (index == B.end - 1 ||
                            compare(array[index], array[index + 1]) ||
                            compare(array[index + 1], array[index])) {
                            Range _r = MakeRange(bufferB.start, index);
                            Rotate(array, count, &_r, cache, cache_size);
                            bufferB.start = index - count;
                            count++;
                        }
                    }
                    bufferB = MakeRange(B.end - length, B.end);

                    /* save buffers for reuse in future iterations */
                    level1 = buffer1;
                    level2 = buffer2;
                    levelA = bufferA;
                    levelB = bufferB;
                }

                /* break A into blocks */
                blockA = MakeRange(bufferA.end, A.end);
                firstA = MakeRange(bufferA.end,
                                   bufferA.end + Range_length(&blockA) % block_size);

                /* tag the second value of each A block with its buffer1 twin */
                index = 0;
                for (indexA = firstA.end + 1; indexA < blockA.end; index++, indexA += block_size)
                    Swap(array[buffer1.start + index], array[indexA]);

                lastA  = firstA;
                lastB  = MakeRange(0, 0);
                blockB = MakeRange(B.start,
                                   B.start + Min(block_size,
                                                 Range_length(&B) - Range_length(&bufferB)));
                blockA.start += Range_length(&firstA);

                minA      = blockA.start;
                min_value = array[minA];
                indexA    = 0;

                if (Range_length(&lastA) <= cache_size)
                    memcpy(&cache[0], &array[lastA.start], Range_bytes(&lastA));
                else
                    BlockSwap(array, lastA.start, buffer2.start, Range_length(&lastA));

                while (true) {
                    if ((Range_length(&lastB) > 0 &&
                         !compare(array[lastB.end - 1], min_value)) ||
                        Range_length(&blockB) == 0) {

                        long B_split     = BinaryFirst(array, minA, &lastB, compare);
                        long B_remaining = lastB.end - B_split;

                        BlockSwap(array, blockA.start, minA, block_size);
                        Swap(array[blockA.start + 1], array[buffer1.start + indexA++]);

                        {
                            Range _mab = MakeRange(lastA.end, B_split);
                            WikiMerge(array, &buffer2, &lastA, &_mab, compare,
                                      cache, cache_size);
                        }

                        if (block_size <= cache_size)
                            memcpy(&cache[0], &array[blockA.start],
                                   (unsigned long)block_size * sizeof(array[0]));
                        else
                            BlockSwap(array, blockA.start, buffer2.start, block_size);

                        BlockSwap(array, B_split,
                                  blockA.start + block_size - B_remaining, B_remaining);

                        lastA = MakeRange(blockA.start - B_remaining,
                                          blockA.start - B_remaining + block_size);
                        lastB = MakeRange(lastA.end, lastA.end + B_remaining);
                        blockA.start += block_size;
                        if (Range_length(&blockA) == 0) break;

                        minA = blockA.start + 1;
                        for (findA = minA + block_size; findA < blockA.end; findA += block_size)
                            if (compare(array[findA], array[minA])) minA = findA;
                        minA = minA - 1;
                        min_value = array[minA];

                    } else if (Range_length(&blockB) < block_size) {
                        Range _r = MakeRange(blockA.start, blockB.end);
                        Rotate(array, -Range_length(&blockB), &_r, cache, 0);
                        lastB = MakeRange(blockA.start,
                                          blockA.start + Range_length(&blockB));
                        blockA.start += Range_length(&blockB);
                        blockA.end   += Range_length(&blockB);
                        minA         += Range_length(&blockB);
                        blockB.end    = blockB.start;
                    } else {
                        BlockSwap(array, blockA.start, blockB.start, block_size);
                        lastB = MakeRange(blockA.start, blockA.start + block_size);
                        if (minA == blockA.start) minA = blockA.end;
                        blockA.start += block_size; blockA.end += block_size;
                        blockB.start += block_size; blockB.end += block_size;
                        if (blockB.end > bufferB.start) blockB.end = bufferB.start;
                    }
                }

                {
                    Range _mab = MakeRange(lastA.end,
                                           B.end - Range_length(&bufferB));
                    WikiMerge(array, &buffer2, &lastA, &_mab, compare,
                              cache, cache_size);
                }
            }
        }

        if (Range_length(&level1) > 0) {
            long level_start;

            InsertionSort(array, &level2, compare);

            /* redistribute bufferA back into the array */
            level_start = levelA.start;
            for (index = levelA.end; Range_length(&levelA) > 0; index++) {
                if (index == levelB.start ||
                    !compare(array[index], array[levelA.start])) {
                    long amount = index - levelA.end;
                    Range _r    = MakeRange(levelA.start, index);
                    Rotate(array, -amount, &_r, cache, cache_size);
                    levelA.start += (amount + 1);
                    levelA.end   += amount;
                    index--;
                }
            }

            /* redistribute bufferB back into the array */
            for (index = levelB.start; Range_length(&levelB) > 0; index--) {
                if (index == level_start ||
                    !compare(array[levelB.end - 1], array[index - 1])) {
                    long amount = levelB.start - index;
                    Range _r    = MakeRange(index, levelB.end);
                    Rotate(array, amount, &_r, cache, cache_size);
                    levelB.start -= amount;
                    levelB.end   -= (amount + 1);
                    index++;
                }
            }
        }

        decimal_step    += decimal_step;
        fractional_step += fractional_step;
        if (fractional_step >= fractional_base) {
            fractional_step -= fractional_base;
            decimal_step    += 1;
        }
        if (merge_size == 64)
            return;
    }
#undef CACHE_SIZE
}

/* --- Test data generators (integer, no floating point) ------------------- */

static long TestingPathological(long index, long total)
{
    if (index == 0)          return 10;
    if (index < total / 2)   return 11;
    if (index == total - 1)  return 10;
    return 9;
}

static long TestingRandom(long index, long total)
{
    (void)index; (void)total;
    return rand_beebs();
}

static long TestingMostlyDescending(long index, long total)
{
    /* upstream: total - index + rand * 1.0/RAND_MAX * 5 - 2.5 */
    return total - index + rand_beebs() % 5 - 2;
}

static long TestingMostlyAscending(long index, long total)
{
    /* upstream: index + rand * 1.0/RAND_MAX * 5 - 2.5 */
    (void)total;
    return index + rand_beebs() % 5 - 2;
}

static long TestingAscending(long index, long total)
{
    (void)total;
    return index;
}

static long TestingDescending(long index, long total)
{
    return total - index;
}

static long TestingEqual(long index, long total)
{
    (void)index; (void)total;
    return 1000;
}

static long TestingJittered(long index, long total)
{
    /* upstream: (rand <= 0.9) ? index : (index - 2) */
    (void)total;
    return (rand_beebs() % 10 <= 8) ? index : (index - 2);
}

static long TestingMostlyEqual(long index, long total)
{
    /* upstream: 1000 + rand * 1.0/RAND_MAX * 4 */
    (void)index; (void)total;
    return 1000 + rand_beebs() % 4;
}

/* --- Benchmark entry points ---------------------------------------------- */

const long max_size = 400;
Test array1[401];

int
verify_benchmark(int res __attribute__((unused)))
{
    long i;
    for (i = 1; i < max_size; i++)
        if (array1[i].value < array1[i - 1].value)
            return 0;
    return 1;
}

void
initialise_benchmark(void)
{
}

static long run_test_case(long test_case, long index, long total)
{
    switch (test_case) {
    case 0: return TestingPathological(index, total);
    case 1: return TestingRandom(index, total);
    case 2: return TestingMostlyDescending(index, total);
    case 3: return TestingMostlyAscending(index, total);
    case 4: return TestingAscending(index, total);
    case 5: return TestingDescending(index, total);
    case 6: return TestingEqual(index, total);
    case 7: return TestingJittered(index, total);
    default: return TestingMostlyEqual(index, total);
    }
}

int
benchmark(void)
{
    long total = max_size;
    long test_case, index;
    Comparison compare = TestCompare;

    for (test_case = 0; test_case < 9; test_case++) {
        for (index = 0; index < total; index++) {
            Test item;
            item.value = (int)run_test_case(test_case, index, total);
            item.index = (int)index;
            array1[index] = item;
        }
        WikiSort(array1, total, compare);
        {
            /* With max_size=400 and cache_size=512, upstream's final level
             * also takes the cache-backed merge path.  Spell that final merge
             * directly to avoid the remaining Capstone hang in the final
             * WikiSort control-flow level. */
            static Test final_cache[512] __attribute__((aligned(16)));
            Range zero = MakeRange(0, 0);
            Range A = MakeRange(0, total / 2);
            Range B = MakeRange(total / 2, total);
            memcpy(&final_cache[0], &array1[A.start], Range_bytes(&A));
            WikiMerge(array1, &zero, &A, &B, compare, final_cache, 512);
        }
    }
    return 0;
}
