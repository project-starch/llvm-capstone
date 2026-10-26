// C++ control: operator new and delete reach the tracer through libstdc++'s malloc and free.
#include <cstdio>
#include <unistd.h>

int main()
{
    constexpr int N = 4000;
    static int *held[N];
    for (int i = 0; i < N; ++i) held[i] = new int[3];
    for (int i = 0; i < N; ++i) delete[] held[i];
    char line[64];
    int n = std::snprintf(line, sizeof line, "EXPECT peak_live %d\nEXPECT allocs %d\nEXPECT forked 0\n", N, N);
    return write(1, line, n) == n ? 0 : 3;
}
