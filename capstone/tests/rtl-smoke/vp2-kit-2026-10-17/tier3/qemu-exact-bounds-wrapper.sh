#!/bin/bash
# Discriminator wrapper: the kit's QEMU with x-capstone-exact-bounds=true appended to the virtual-profile -cpu string.
args=(); prev=""
for a in "$@"; do
  if [ "$prev" = "-cpu" ] && [[ "$a" == rv64,*x-capstone-u-mode=true* ]] && [[ "$a" != *x-capstone-exact-bounds* ]]; then a="$a,x-capstone-exact-bounds=true"; echo "$(date +%T) -cpu $a" >> /tmp/capstone/vp2-prep/v6-fix-exactkit.wrapper.log; fi
  args+=("$a"); prev="$a"
done
exec /tmp/capstone/vp2-prep/v6-fix/out/kit/qemu-system-riscv64 "${args[@]}"
