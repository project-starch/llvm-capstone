#!/usr/bin/env bash
# Regenerate the Excel view of the bug table and (optionally) push it to Drive.
#
# The CSV in this directory is the SOURCE OF TRUTH and is what gets committed:
# it diffs, it reviews, it merges. The .xlsx is a derived view for people who
# read it in a spreadsheet, it lives in Google Drive, and it is gitignored here
# precisely so a binary blob never becomes the thing everyone edits.
#
#   usage:  bash make-xlsx.sh [--upload]
#
# Needs the venv at ~/.venvs/xlsx (python3 -m venv ~/.venvs/xlsx && pip install openpyxl).
# Drive target is the KISP Shared/Capstone folder via the existing rclone remote.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CSV="$HERE/sqlite-cves.csv"
OUT="${TMPDIR:-/tmp}/sqlite-cves.xlsx"
VENV="$HOME/.venvs/xlsx/bin/python"
REMOTE="gdrive:KISP Shared/Capstone/"

[[ -f "$CSV" ]] || { echo "no CSV at $CSV" >&2; exit 1; }
[[ -x "$VENV" ]] || { echo "no venv python at $VENV -- see header" >&2; exit 1; }

"$VENV" - "$CSV" "$OUT" <<'PY'
import csv, sys
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

csv_path, out = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(csv_path, newline='', encoding='utf-8')))
if len(rows) < 2:
    sys.exit("refusing to build a spreadsheet from an empty table")

wb = Workbook(); ws = wb.active; ws.title = "sqlite-bugs"
for r in rows:
    ws.append([int(v) if v.isdigit() else v for v in r])
hdr, fill = Font(bold=True), PatternFill("solid", fgColor="DDDDDD")
for c in ws[1]:
    c.font, c.fill = hdr, fill
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions
for ci in range(1, ws.max_column + 1):
    w = max(len(str(ws.cell(r, ci).value or "")) for r in range(1, ws.max_row + 1))
    ws.column_dimensions[get_column_letter(ci)].width = min(w + 2, 60)
wb.save(out)

# self-check: a spreadsheet that silently lost rows is worse than none
back = load_workbook(out).active
assert back.max_row == len(rows), f"{back.max_row} rows written, {len(rows)} expected"
assert back.max_column == len(rows[0]), "column count drifted"
print(f"{out}: {back.max_row} rows x {back.max_column} cols")
PY

if [[ "${1:-}" == "--upload" ]]; then
  rclone copy "$OUT" "$REMOTE" && echo "uploaded to $REMOTE"
fi
