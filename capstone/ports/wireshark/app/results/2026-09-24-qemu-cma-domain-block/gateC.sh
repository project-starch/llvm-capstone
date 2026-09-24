run() { echo "RUN $1"; /mnt/host/gateload.user /mnt/host/$1.dom; echo "EXIT $1 rc=$?"; dmesg | grep -E "Domain block|Failed to allocate|does not survive|beyond the buddy|tot_size" | tail -n 3 | sed "s/^/DMESG $1: /"; }
echo "MODULE-MD5 $(md5sum /capstone.ko)"
run small; run smalldecl; run big128; run edge64
echo GATEC-END
