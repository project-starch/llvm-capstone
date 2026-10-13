#!/usr/bin/env bash
# MREV over a node with CDERIVE children, on capstone-qemu's virtual profile; see README.md.
set -uo pipefail
main() {
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
if [[ ${CAPSTONE_QEMU_LOCK_HELD:-0} != 1 ]]; then
    exec env CAPSTONE_QEMU_LOCK_HELD=1 flock -w "${CAPSTONE_QEMU_LOCK_WAIT:-3600}" \
        "$CAPSTONE_QEMU_LOCK" bash "$here/run.sh" "$@"
fi
qroot=${CAPSTONE_QEMU_SRC:-$CAPSTONE_REPO_ROOT/capstone/capstone-qemu}
qemu=${VIRTUAL_CAPSTONE_QEMU_BINARY:-$qroot/build/qemu-system-riscv64}
work=$(mktemp -d "$CAPSTONE_TMP_ROOT/mrev-over-derived.XXXXXX")
echo "qemu $qemu sha256 $(sha256sum "$qemu" | cut -c1-16); clang $(command -v clang) $(clang --version | head -1)" | tee "$work/identity.txt"
# arm: defines, the gate case id that selects the expected cause, allocated nodes, expected marker.
#   dup        cause 25 at the load after REVOKE   8 bootstrap + CDERIVE + MREV
#   norevoke   cause 3 at the ebreak               the same, without the REVOKE
#   inplace    cause 26 at the MREV                8 + CDERIVE; MREV allocates nothing
#   the last three are dup with a wrong PC, cause or node count, and must FAIL
arms=("dup:-:1:10:PASS" "norevoke:-DNO_REVOKE=1:0:10:PASS" "inplace:-DIN_PLACE=1:11:9:PASS"
      "wrongpc:-DWRONG_PC=1:1:10:FAIL" "wrongcause:-:11:10:FAIL" "wrongnodes:-:1:9:FAIL")
failed=0 total=0
for spec in "${arms[@]}"; do
    IFS=: read -r arm define case_id nodes marker <<< "$spec"
    for layout in flat paged; do
        total=$((total+1))
        flags=()
        [[ $define == - ]] || flags+=("$define")
        [[ $layout != paged ]] || flags+=(-DPAGED_NODES=1)
        elf=$work/$arm-$layout.elf
        if ! clang --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -nostdlib \
            -fuse-ld=lld "${flags[@]}" -DSUBLET_CASE="$case_id" -DSUBLET_NODE_COUNT="$nodes" \
            "-DSUBLET_PROGRAM=\"$here/program.inc\"" \
            -Wl,-T,"$qroot/tests/virtual-capstone-runtime/link.ld" \
            "$qroot/tests/sublet-lifetimes/probe.S" -o "$elf"; then
            echo "FAIL $arm $layout (build)"; failed=$((failed+1)); continue
        fi
        status=0
        timeout 10s "$qemu" -M virt -cpu rv64,x-capstone-u-mode=true \
            -bios none -smp 1 -nographic -semihosting \
            -device "loader,file=$elf" > "$elf.log" 2>&1 || status=$?
        expected=0
        [[ $marker == PASS ]] || expected=1
        if [[ $status == "$expected" ]] && grep -qx "VIRTUAL-CAPSTONE-M1:$marker" "$elf.log"; then
            echo "PASS $arm $layout"
        else
            echo "FAIL $arm $layout (rc=$status, $elf.log)"; failed=$((failed+1))
        fi
    done
done
echo "$((total-failed))/$total passed; logs: $work"
[[ $failed == 0 ]]
}
main "$@"; exit
