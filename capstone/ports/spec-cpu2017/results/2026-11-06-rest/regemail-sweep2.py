"""Twin (kit-profq) sweep for re/reg_email under the arena cache: perl-ac failed it on silicon on 6 Nov (test.out:
"re/reg_email ... FAILED--no leader found"), while its QEMU gate passed. The October defect of the same test was an
allocation-history-dependent realloc exactness trap, which env-size pads reproduced on the twin (perl-regemail/
FINDINGS.md). Images: HF = the heapfix production image (perf/rest -devmk, control), AC = the board's perl-ac image,
BEST = c3s1. Steps per image: harness invocation with PAD of 0..2048 bytes, and the board-path mimic."""
import os, sys
W = os.environ["CAPSTONE_WORK"]
sys.path.insert(0, W + "/wt-spec/capstone/bug-corpora/tools")
import virtualvm
H = W; R = H + "/perf/rest/stages"; B = H + "/perf/best/stages"
out = sys.argv[1]
b = virtualvm.Batch(out + "/batch")
IMG = {"HF": b.put("hf.img", R + "/500.perlbench_r-test-capstone-lifetimes-devmk/stage/bin/capstone-lifetimes/perlbench_r_base.capstone-lifetimes"),
       "AC": b.put("ac.img", R + "/500.perlbench_r-test-capstone-lifetimes-devmk-ac/stage/bin/capstone-lifetimes/perlbench_r_base.capstone-lifetimes"),
       "BEST": b.put("best.img", B + "/500.perlbench_r-test-capstone-lifetimes-devmk-c3s1/stage/bin/capstone-lifetimes/perlbench_r_base.capstone-lifetimes")}
inp = b.put("in", R + "/500.perlbench_r-test-capstone-lifetimes-devmk-ac/stage/in")
V = "/mnt/vm/capstone-vexec"
PADS = [0, 8, 64, 256, 512, 1024, 2048]
def summ(tag):
    return (f"echo \"@@R {tag} rc=$rc out-lines=$(wc -l < /tmp/o) ok=$(grep -c '^ok' /tmp/o) notok=$(grep -c '^not ok' /tmp/o)"
            f" plan=$(grep -a '^1\\.\\.' /tmp/o | tail -1) :: $(dmesg | grep 'capstone_vm: bench' | tail -1 | sed 's/.*capstone_vm: //')\"; "
            f"grep -a 'capstone-exec:\\|CAPSTONE_VM_STATS\\|Segmentation\\|panic\\|Out of memory\\|cause' /tmp/e | grep -av CAPSTONE_VM_ADD_OK | head -8 | sed 's|^|@@E {tag}: |'; "
            f"tail -c 600 /tmp/e | sed 's|^|@@T {tag}: |'")
def prep(launcher, img):
    return (f"rm -rf /tmp/sr /tmp/sw && mkdir -p /tmp/sr /tmp/sw && cp -a {inp}/. /tmp/sr/ && "
            f"printf '#!/bin/sh\\nexec \"%s\" --application-argv \"%s\" -- \"$0\" \"$@\"\\n' {launcher} {img} > /tmp/sw/perl && "
            f"chmod 755 /tmp/sw/perl && cd /tmp/sr/t")
def harness(tag, img, pad):
    pre = f"PAD={'x' * pad} " if pad else ""
    b.step(tag, f"{prep(V, img)} && export CAPSTONE_VM_STATS=1 PERL_DESTRUCT_LEVEL=2 PERL_DL_NONLAZY=1 EMXSHELL=sh && "
                f"{pre}../../sw/perl -I.. -MTestInit re/reg_email.t > /tmp/o 2> /tmp/e; rc=$?; {summ(tag)}")
BD = "/test-domains/vbatch"
def boardmimic(tag, img):
    tgt = f"{BD}/spec/bin/capstone-lifetimes/perlbench_r_base.capstone-lifetimes"
    b.step(tag, f"mkdir -p {BD}/spec/bin/capstone-lifetimes && cp {V} {BD}/capstone-vexec && cp {img} {tgt} && "
                f"{prep(BD + '/capstone-vexec', tgt)} && export CAPSTONE_VM_STATS=1 SPEC_VEXEC={BD}/capstone-vexec PERL_DESTRUCT_LEVEL=2 "
                f"PERL_DL_NONLAZY=1 EMXSHELL=sh && ../../sw/perl -I.. -MTestInit re/reg_email.t > /tmp/o 2> /tmp/e; rc=$?; {summ(tag)}")
for name in ("AC", "BEST", "HF"):
    boardmimic(f"{name}-BM", IMG[name])
    for L in PADS: harness(f"{name}-P{L}", IMG[name], L)
runs, transcript, completed = b.execute(H + "/spec/kit-profq", out + "/guest", timeout=6 * 3600)
print("completed", completed, transcript)
for n, (text, res) in runs.items():
    print("=====", n, res)
    for line in (text or "").splitlines():
        if line.startswith("@@"): print(line[:400])
