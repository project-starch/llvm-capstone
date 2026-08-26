# Finding the "a capability is an integer" bug class, long-term

Nine instances of ONE bug in three days, each found by a different accident. This
is what the class actually is, which instruments can see it, what each one costs,
and what the measured coverage gaps are. The recommendation at the end is ordered
by bugs-found-per-hour, measured rather than guessed.

## What the class is, precisely

`VTCheriCapability` sets `isCheriCapability = true` and nothing else, so `c128`
answers **false** to `isInteger()`, `isFloatingPoint()` and `isVector()` alike. It
is none of the above. That is correct, and it is why the generic asserts started
firing at all.

The bugs are therefore all one shape: **code that tests "not a vector" or "not
floating point" and concludes "integer"**, or that never tests and assumes it.

    getSetCCResultType        if (!VT.isVector()) return getPointerTy(DL);
    SELECT_CC combine         no test at all, then arithmetic on VT
    convertSelectOfConstants  return true, unconditionally

While `i128` WAS the capability type, every one of these got a true answer to
`isInteger()` and the wrong behaviour went through silently. The type change did
not create the bugs; it made them visible.

`getPointerTy(DL)` deserves naming separately. On RISC-V it is i64 and doubles as
"an XLen register", so upstream code uses it for both meanings. Here the two
diverge, and 21 sites in the backend use it. Most are genuine pointer contexts;
`getSetCCResultType` was not.

## The instruments, and what each can and cannot see

| Instrument | Sees | Blind to | Cost |
|---|---|---|---|
| Generic LLVM asserts | crashes, on executed paths only | anything that produces wrong code without asserting | free, already on in a debug build |
| **`-verify-machineinstrs`** | **invalid machine code: wrong operand class, wrong register** | a wrong DECLARATION (DROP declared GPR, so a GPR in it was "valid") | free, ~15% compile time |
| `scan-tag-stripped-caps.py` | one shape: `mv rd, rs` feeding a capability use | every other way to lose a tag | seconds, needs a disassembly |
| Execution of self-checking work | wrong ANSWERS | anything not on the executed path | minutes to hours, QEMU |
| Differential vs upstream riscv64 | wrong code for capability-free IR | anything involving capabilities | not built yet |

The asymmetry that matters: **asserts and the verifier catch crashes and invalid
code; only execution and differential testing catch a silently wrong answer.**
Every bug found so far by asserts was a crash. The original report that started
all of this -- `unsigned __int128 a + b` compiling to `cincoffset` -- was silent,
and no static check would have caught it.

## The measured coverage gaps

**The verifier is not run on the ABI that ships.** Of 58 Capstone lit tests, 34
run `-verify-machineinstrs` and 2 use `gp-captable`/`gp-free`. The intersection is
EMPTY. No build script in the project passes the flag either. Turning it on over
musl fired on 60 of 60 sampled files and produced two real bugs in an hour: an
`SD` on a capability register with a memory operand claiming twice what it wrote,
and a `CIncOffset` writing an X register whose untagged result the next
instruction then used as a capability base. The second faults at run time.

**The largest C corpus available is musl, and it was not being compiled.** 1361
surveyable files. Running it found three backend assertions in one pass. The
in-tree corpora are an order of magnitude smaller: beebs 56 files, sqlite 47,
rv8 14, coremark 8.

**Bounded audit surfaces**, for when the corpus stops yielding: 27 sites in the
backend test `!VT.isVector()`; 40 overridden TargetLowering hooks take or return
a type; 21 uses of `getPointerTy`. Each has one question -- what should this
answer for a capability -- and a definite endpoint.

## Order, by measured yield

1. **Run `-verify-machineinstrs` in the musl survey and in every build script.**
   Three bugs on first contact, ~15% compile time. It is the cheapest instrument
   by a wide margin because it already exists and nobody was running it. DONE:
   it is now one of the survey's own flags, which the build script and any
   experiment inherit through `--print-flags`.

   **Take the flags from the tool, never by hand.** My first sweep transcribed
   them and reported "1530 files, 0 errors" while the third bug was sitting in
   src/time/timer_create.c. The survey's `--print-flags` exists precisely so
   there is nothing to transcribe; I did not use it, and the number I published
   was wrong.
2. **Keep musl in the loop as the compile corpus.** It found three asserts and
   surfaced both verifier bugs. It is 1361 files against beebs's 56.
3. **Pair every new ABI flag with a verifier test.** The hole here was not that
   the verifier was missing but that the two test dimensions never crossed.
   `cap-gp-free-frame-verifier.ll` is three lines and needs only a call.
4. **Build the differential oracle against riscv64.** It is the only instrument
   that would catch a silently wrong ANSWER for capability-free code, which is
   the dangerous half of this class and is currently unguarded. Feasibility is
   established: the same IR compiled for both targets produced identical
   instruction sequences in every hand check during this work.
5. **Then the bounded audits**, in order of density: the 27 `!isVector()` sites,
   then the 40 type hooks.

## What NOT to do

Do not widen `scan-tag-stripped-caps.py` to chase shapes. It is per-block and
per-register by design, and today's `CIncOffset` bug had no `mv` in it at all --
the verifier saw it, the scanner could not have. Its docstring already names the
ceiling.

Do not treat a clean corpus as evidence without a positive control. The scanner
carried eight documented "hits" as proof it worked; they were false positives
from checking the wrong operand index, and the real operand had never been
checked. A control built from the instruction's operand list would not have had
that problem; one harvested from an observed failure did.
