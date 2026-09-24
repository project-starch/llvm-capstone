/* ISSUES C-65: musl-capstone's pthread_cond_t cannot hold its own fields. It is 48 bytes, and
 * musl's _c_tail (__u.__p[5]) is read at +80. On the heap, level0 puts the next block's header
 * right after the object, and a free block's `free` flag, 1, lies at +80: broadcast then walks from
 * the integer 1. Natively (glibc) the broadcast returns. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
int main(void)
{
	pthread_cond_t *c = malloc(sizeof *c);
	printf("sizeof(pthread_cond_t) = %zu\n", sizeof *c);
	fflush(stdout);
	pthread_cond_init(c, 0);
	pthread_cond_broadcast(c);
	printf("broadcast returned\n");
	return 0;
}
