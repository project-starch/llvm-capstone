# CMASan (S&P 2025) and what it means for this project

**"CMASan: Custom Memory Allocator-aware Address Sanitizer"**, IEEE S&P 2025.
<https://yonghwi-kwon.github.io/data/cmasan_sp25.pdf>, tool at
<https://github.com/S2-Lab/CMASan>. (Cited by title and URL only; this repository
does not carry personal names.)

## Why it matters here

It is the same problem statement as our nested-allocator work, attacked from the
other side. Their sentence for it: ASan "only tracks objects allocated by standard
memory allocators", so a custom memory allocator (CMA) that carves one arena into
many objects gets **one redzone around the arena and none between the objects**.
That is the software-sanitizer statement of what our disassembly and QEMU pairs
measured about the hardware: one capability over the whole region, and nothing to
tell two sub-allocations apart.

They fix it in the compiler. We are asking what the *silicon* can see. The two are
complementary and the paper is the stronger citation for the premise, because it is
peer-reviewed, quantified, and not ours.

## The numbers worth quoting

From their Table 1, MicroPython specifically:

| | value |
|---|---|
| CMA objects | 839,561 |
| base chunks | 957 |
| freed CMA objects | 110,415 |
| load/store checks on CMA objects | 467,351,689 (9.79% of all checks) |
| CMA pattern | **Arena + Recycler** |

**957 base chunks carry 839,561 objects.** That ratio is the nested-allocator gap
as a number: ASan's redzones, and our capability bounds, are placed 957 times while
the program creates 839,561 objects that could be out of bounds with respect to
each other. It is independent quantitative support for
`benchmarks/micropython/temporal-corpus/evidence/heap-bounds-model.s` and the two
matched pairs in `tests/runtime-qemu/silicon-ladder/`.

Other figures: CMASan costs 9.63% over ASan on average (3.15% on MicroPython), and
ASan detects **none** of the 19 bugs they report -- every row of their Table 5 has
a cross in the ASan column. Our `asan-blindness-2026-08-17.txt` shows the same
thing with a twenty-line C model; this is the large-scale version.

They also name MicroPython's `gc_realloc` as the worked example of in-place realloc,
the case where the Shim approach (swap the CMA for the standard allocator) breaks
because there is no 1:1 mapping to `realloc`.

## Their MicroPython bugs against our corpora

Their Table 5 lists nine MicroPython findings. **All nine are already in our
corpora**, found independently:

| their row | ours |
|---|---|
| CVE-2023-7152 | `MPY-T01` (#12887) |
| CVE-2023-7158 | `MPY-S03` (#13007) |
| CVE-2024-8946 | `MPY-S09` (#13006) |
| CVE-2024-8947 | `MPY-T02` (#13283) |
| CVE-2024-8948 | `MPY-S02` (#13041) |
| Issue #13004 | `MPY-S10` |
| Issue #13046 | `MPY-S08` |
| Issue #13220 | `MPY-S04` |
| Issue #13428 | `MPY-S07` |

Three of those CVE identifiers were not in our corpora as CVEs, only as issue
numbers; they are cross-referenced now, along with the fix commits the CVE records
carry, which the GitHub timeline API had not given us.

Separately checked: NVD lists **six** MicroPython CVEs in total, and all six are in
our corpora. The CVE space is exhausted; anything further has to come from issues.

## What this does NOT say, and where we differ

- CMASan detects bugs; it says nothing about whether a capability machine would.
  Their result is that a *sanitizer build* can be made to see inside a CMA. Ours is
  that an *unmodified production binary* on capability hardware cannot.
- Their scope is explicitly "memory bugs on objects allocated by applications
  through CMAs"; bugs *inside* the allocator are out of scope for them. Several of
  our rows (`MPY-T29`, `MPY-T28`) are allocator-internal.
- They report a 32.32% false-positive rate on MicroPython before their avoidance
  techniques (298 of 922 tests). Worth knowing before anyone proposes running their
  tool as an oracle for our corpus.

## If someone wants to use it

The tool is open source and builds on Clang 15.0.6. It would give a second,
independent verdict per corpus row on the host, which is exactly the oracle our
`stock_behaviour` column is guessing at for the rows measured only from issue text.
That is a real piece of work, not a quick check: it needs its CodeQL-based CMA
identification run over MicroPython and a manual categorisation pass the paper
budgets at roughly ten minutes per application.
