#!/bin/bash
mkdir -p $3; echo "grade: CONFIRMED (stub)" > $3/grade.txt; exit ${BARE_RC:-0}
