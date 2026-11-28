#!/bin/bash
# ns-build.sh: rebuild the PHP image at the PATH the measured image was built at, without writing there.
# A private mount namespace binds this work directory's php-diag over the first pass's php-meas for this process
# tree alone, so the compiler sees the same source paths (422 of them are embedded in the image) and every write
# lands in php-diag. The sentinel check refuses to build if the bind is not in place.
source $HOME/crenew-step4/corpus/work-fix-php/env.sh
exec unshare -Urm bash -c '
  set -e
  mount --bind "$W/php-diag" "$A/php-meas"
  [ -e "$A/php-meas/.fix-php-sentinel" ] || { echo "bind not in place, refusing" >&2; exit 3; }
  export PHP_VIRTUAL_OUT=$A/php-meas
  cd $TREE
  "$@"
' ns-build "$@"
