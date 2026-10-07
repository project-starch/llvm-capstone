# capstone/bug-corpora/cpython/pymalloc-repros in the virtual address space

Arm `virtual`, run 20261007-virtual. Status **PASS**:
20 of 20 cases measured, 0 of
0 controls held. Verdicts: 20 silent.

Every case is one Linux process under the virtual launcher `capstone-vexec`, so
a capability fault ends that process and the rest of the corpus keeps running --
the whole corpus is one boot. `matrix.tsv` has one line per row; `inputs.json`
carries the sha256 of every image, of the gate script, of each staged resource
and of the QEMU binary, launcher, module and kernel that ran them.

A verdict means:

| verdict | what it says |
|---|---|
| `detected` | a Capstone capability fault, causes 24-30, with the cause named |
| `trap` | the process stopped on something else -- an illegal instruction, an access or page fault. A stop, but not the capability mechanism answering |
| `silent` | the case ran its own pre-defect marker and completed |
| `control-failure` | the case refused its own setup; never a verdict about the defect |
| `harness` | the case did not run, or faulted before announcing itself. Not a measurement |
| `timeout` | killed at the per-case limit. Not a measurement |

`attributed` is filled only where the corpus's labelled probe is an external
symbol: the fault's pc is compared against that symbol's extent resolved from
the image that ran, shifted by the load address the launcher published. A `--`
means the corpus's probe is `static`, so no attribution is claimed.

Reproduce with `tools/build-virtual-*.py` and `tools/run-virtual-corpus.py`; the
lane document is `capstone/docs/plans/bug-corpora-virtual-address-space.md`.
