#!/usr/bin/env bash
# The contrib extensions the sql-repros corpus triggers, for a CheriBSD purecap
# arm: built against an already-configured purecap build tree and staged beside
# the backend that will dlopen them.
#
# Four cases need one each -- 03 ltree, 07 pg_trgm, 08 fuzzystrmatch,
# 09 pgcrypto -- and a case whose extension is absent does not fail loudly. The
# statement CREATE EXTENSION <name> errors on the missing control file, the rest
# of the trigger never runs, and the arm records a quiet pass. That is a control
# failure, not a negative result, and it is what the 2026-10-05 Capstone run of
# case 03 recorded before this was noticed.
#
#   PG_PURECAP_BUILD=<configured build tree> \
#   PG_PURECAP_STAGE=<prefix staged into the guest, e.g. .../usr/local/pgsql> \
#   PG_PURECAP_SRC=<the 17.5 source that tree was configured from> \
#     bash build-cheribsd-contrib.sh
#
# pgcrypto needs -lcrypto. The tree is configured without OpenSSL, so its LIBS
# carry no -lcrypto and the module's own Makefile filters for one that is not
# there; SHLIB_LINK below supplies it directly. That is enough because pgcrypto
# is dlopened: the module takes its own DT_NEEDED on libcrypto.so.30 and the
# backend never has to know about OpenSSL. CheriBSD's OpenSSL 3.0.16 exports
# EVP_cast5_cbc and the other 45 symbols the module needs, all checked below.
set -euo pipefail
: "${PG_PURECAP_BUILD:?set PG_PURECAP_BUILD}"
: "${PG_PURECAP_STAGE:?set PG_PURECAP_STAGE}"
: "${PG_PURECAP_SRC:?set PG_PURECAP_SRC}"
MODULES=${PG_PURECAP_MODULES:-"ltree pg_trgm fuzzystrmatch pgcrypto pgcorpus_reach"}
JOBS=${JOBS:-8}

[[ -f $PG_PURECAP_BUILD/config.status ]] || { echo "not a configured tree: $PG_PURECAP_BUILD" >&2; exit 2; }
[[ -d $PG_PURECAP_STAGE/lib && -d $PG_PURECAP_STAGE/share/extension ]] \
  || { echo "not a staged prefix: $PG_PURECAP_STAGE" >&2; exit 2; }
# A vpath build reaches its sources through symlinked Makefiles. If the source
# tree has been moved or removed every one of them dangles, and make reports
# only "No rule to make target 'Makefile'", which does not say why.
[[ -f $PG_PURECAP_SRC/contrib/ltree/Makefile ]] \
  || { echo "source tree missing or not 17.5: $PG_PURECAP_SRC" >&2; exit 2; }

# The corpus's own module is kept in the repository, not in the pinned tarball,
# so it is copied into the source tree the build tree reaches through its
# symlinked Makefiles. Cases 05 and 06 are reached by a C caller because no SQL
# statement can reach them, and a module is how a C caller runs in the backend.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $MODULES == *pgcorpus_reach* ]]; then
  mkdir -p "$PG_PURECAP_SRC/contrib/pgcorpus_reach"
  cp "$HERE"/reach-module/* "$PG_PURECAP_SRC/contrib/pgcorpus_reach/"
  mkdir -p "$PG_PURECAP_BUILD/contrib/pgcorpus_reach"
  ln -sf "$PG_PURECAP_SRC/contrib/pgcorpus_reach/Makefile" \
    "$PG_PURECAP_BUILD/contrib/pgcorpus_reach/Makefile"
fi

# Generated headers live in the build tree, not the source tree, and a module
# compiled without them stops at storage/lwlocknames.h.
make -C "$PG_PURECAP_BUILD/src/backend" generated-headers > /dev/null

for m in $MODULES; do
  link=()
  [[ $m == pgcrypto ]] && link=(SHLIB_LINK=-lcrypto)
  make -C "$PG_PURECAP_BUILD/contrib/$m" -j"$JOBS" "${link[@]}" "$m.so" > /dev/null
  so=$PG_PURECAP_BUILD/contrib/$m/$m.so
  [[ -f $so ]] || { echo "$m: no $m.so" >&2; exit 2; }
  install -m 0755 "$so" "$PG_PURECAP_STAGE/lib/$m.so"
  install -m 0644 "$PG_PURECAP_SRC/contrib/$m/$m.control" "$PG_PURECAP_STAGE/share/extension/"
  install -m 0644 "$PG_PURECAP_SRC/contrib/$m"/"$m"--*.sql "$PG_PURECAP_STAGE/share/extension/"
  echo "$m: staged $(basename "$so") and $(ls "$PG_PURECAP_SRC/contrib/$m"/"$m"--*.sql | wc -l) version scripts"
done

# Every case reads its extension through CREATE EXTENSION, so the control file
# matters as much as the library. Fail here rather than let an arm record a
# quiet pass for a trigger whose first statement did not run.
for m in $MODULES; do
  [[ -f $PG_PURECAP_STAGE/lib/$m.so ]] || { echo "$m.so did not stage" >&2; exit 2; }
  [[ -f $PG_PURECAP_STAGE/share/extension/$m.control ]] || { echo "$m.control did not stage" >&2; exit 2; }
done
echo "staged into $PG_PURECAP_STAGE: $MODULES"
