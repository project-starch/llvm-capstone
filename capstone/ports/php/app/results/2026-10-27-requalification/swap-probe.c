#include <stdio.h>
#include <string.h>
#include <stdlib.h>
/* Does swapping two pointers through int-sized (then char-sized) stores keep them usable?
   That is PHP 5.0.0 Zend/zend_qsort.c:_zend_qsort_swap on a Bucket** array. Modes: int, char, memcpy. */
static void swap_int(void *a, void *b, size_t siz) {
    int *pa = a, *pb = b; size_t i; for (i = sizeof(int); i <= siz; i += sizeof(int)) { int t = *pa; *pa++ = *pb; *pb++ = t; }
}
static void swap_char(void *a, void *b, size_t siz) {
    char *pa = a, *pb = b; size_t i; for (i = 0; i < siz; i++) { char t = pa[i]; pa[i] = pb[i]; pb[i] = t; }
}
static void swap_memcpy(void *a, void *b, size_t siz) {
    char tmp[64]; memcpy(tmp, a, siz); memcpy(a, b, siz); memcpy(b, tmp, siz);
}
int main(int argc, char **argv) {
    int *x = malloc(sizeof *x), *y = malloc(sizeof *y); *x = 11; *y = 22;
    int *arr[2] = { x, y };
    const char *mode = argc > 1 ? argv[1] : "int";
    printf("mode=%s before=%d,%d\n", mode, *arr[0], *arr[1]); fflush(stdout);
    if (!strcmp(mode, "int")) swap_int(&arr[0], &arr[1], sizeof arr[0]);
    else if (!strcmp(mode, "char")) swap_char(&arr[0], &arr[1], sizeof arr[0]);
    else swap_memcpy(&arr[0], &arr[1], sizeof arr[0]);
    printf("swapped, tag0=%d tag1=%d\n", (int)__builtin_capstone_cap_get_tag(arr[0]), (int)__builtin_capstone_cap_get_tag(arr[1])); fflush(stdout);
    printf("after=%d,%d\n", *arr[0], *arr[1]);
    return 0;
}
