"""Append rows to 결과_원본 of the RSS2027 workbook in place (same columns as the earlier rows:
실험, 벤치마크, 태스크, 손, 방법, 성공, 횟수, 일자, 출처, 회차). A row whose (실험, 벤치마크, 태스크, 손, 방법, 회차) exists is replaced.
  uv run --with openpyxl python xlsx_rows.py <workbook.xlsx> <rows.json: {"rows": [[...], ...]}>
"""
import json
import sys

from openpyxl import load_workbook
from openpyxl.styles import Font

path, spec = sys.argv[1:3]
rows = json.load(open(spec))["rows"]
wb = load_workbook(path)
w = wb["결과_원본"]
have = {tuple(str(c.value) for c in r[:5]) + (str(r[9].value),): r for r in w.iter_rows(min_row=2) if r[0].value}
for row in rows:
    old = have.get(tuple(map(str, row[:5])) + (str(row[9]),))
    if old is not None:
        for c in range(5, 10):
            old[c].value = row[c]
        continue
    r = w.max_row + 1
    for c, v in enumerate(row, 1):
        w.cell(r, c, v).font = Font(name="Arial", size=10, color="0000FF" if c in (6, 7, 10) else "000000")
    w.cell(r, 11, "위치 5 cm").font = Font(name="Arial", size=10)
wb.save(path)
print(len(rows), "rows")
