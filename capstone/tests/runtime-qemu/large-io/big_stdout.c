/* One stdout line longer than a payload region, then an end marker. run.sh runs
 * this with the host's stdout redirected to a file on the 9p share and compares
 * the file with what must be there: byte i of the long line is 'a' + i % 26. The
 * domain's stdout is flushed at exit in region-sized WRITE_STDOUT rounds, so the
 * first round hands the host 4096 bytes, well past 9p's 1024-byte zero-copy
 * threshold. */
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
