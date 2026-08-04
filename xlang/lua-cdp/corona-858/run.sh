#!/usr/bin/env bash
# Corona/Solar2D #858 is BLOCKED — see CASE.md. There is no apt/prebuilt Solar2D
# SDK (apt-cache policy solar2d corona corona-sdk -> 0 candidates) and a
# from-source engine build (librtt + Box2D + platform runtime) is out of scope.
# This marker keeps the per-case harness uniform and exits 2 (BLOCKED, not FAIL).
set -euo pipefail
echo "BLOCKED: Corona/Solar2D #858 not reproduced."
echo "  reason: full Solar2D/Corona native engine + Box2D required; no apt/prebuilt"
echo "          SDK, from-source mobile-engine build out of scope."
if apt-cache policy solar2d corona corona-sdk 2>/dev/null | grep -q 'Candidate: [^(]'; then
  echo "  NOTE: an SDK package now appears in apt — revisit this BLOCKED status." >&2
fi
echo "  free-site: PhysicsWorld::StopWorld -> Rtt_DELETE(fWorld)  (Rtt_PhysicsWorld.cpp:209)"
echo "  use-site:  PhysicsJoint::Finalizer -> UserdataWrapper::Dereference() on freed b2Joint (Rtt_PhysicsJoint.cpp:46)"
echo "  see CASE.md / boundary.md / evidence.txt (source-quoted from the real tree)."
exit 2
