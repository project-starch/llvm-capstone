/* One stdout line of 5000 bytes, then an end marker. ../run-delegated-probes.py
 * runs this with the launcher's stdout redirected to a file on the 9p share and
 * compares the file with what must be there: byte i of the long line is
 * 'a' + i % 26. musl's stdout buffer is 8192 bytes, so both lines leave in one
 * write at exit, 5016 bytes, and the launcher's write(2) hands 9p more than its
 * 1024-byte zero-copy threshold, from its bounce buffer. */
#include <stdio.h>

#define LINE 5000

int main(void)
{
	static char line[LINE + 2];
	for (int i = 0; i < LINE; i++)
		line[i] = (char)('a' + i % 26);
	line[LINE] = '\n';
	fputs(line, stdout);
	puts("BIG-STDOUT-END");
	return 0;
}
