#!/bin/bash
# chain-board stub: create an out dir per list line, log BEGIN/END
while read -r s a b t; do case $s in ''|'#'*) continue;; esac; n=$(basename $s); [ -e $3/$n ] && continue; mkdir -p $3/$n; echo "$(date +%T) BEGIN $n" >> $3/chain-b$1.log; echo "$(date +%T) END $n rc=0" >> $3/chain-b$1.log; done < $2
