"""Print one space's row of the artifact "전체 지표" table next to the reference spaces (2026-11-19).
  python full_row.py --work <run_all work dir> --name <name> --bench dv_right [--ref reference_table.json]"""
import argparse, json, math, os

COLS = [("r1", "R@1 %"), ("mrank", "중앙순위"), ("prec", "정밀도 R@1 %"), ("bund", "손간 해상도 %"),
        ("cross", "교차 재구성 mm"), ("demo_line", "데모 교차 °"), ("pinch_line", "핀치 교차 °"), ("rand_line", "무작위 교차 °"),
        ("demo_self", "데모 자기 mm"), ("rand_self", "무작위 자기 mm"), ("pself", "쌍 자기 mm"),
        ("res_pct", "해상도 %"), ("res_mm", "해상도 mm"), ("cont", "연속성 mm")]


def row(work, name, bench):
    base = json.load(open(f"{work}/{bench}/space_{name}.json"))["mean"]
    dp = json.load(open(f"{work}/{bench}_demopair/space_{name}.json"))["mean"]
    pm = json.load(open(f"{work}/pair_metrics_{name}.json"))
    return {"r1": pm.get("r1"), "mrank": pm.get("mrank"), "prec": pm.get("prec"), "bund": pm.get("bund"),
            "cross": dp.get("m2_cross_tip_mm"), "pself": dp.get("pair_self_tip_mm"),
            "demo_self": base.get("m1_teleop_self_tip_mm"), "rand_self": base.get("m5_random_self_tip_mm"),
            "demo_line": base.get("m3_teleop_line_dir_deg"), "pinch_line": base.get("m6_pinch_line_dir_deg"),
            "rand_line": base.get("m6_random_line_dir_deg"), "res_pct": base.get("m10_res_rel_pct"),
            "res_mm": base.get("m10_res_tip_mm"), "cont": base.get("m7_cont_tip_mm")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--name", required=True); ap.add_argument("--bench", default="dv_right")
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_table.json"))
    a = ap.parse_args()
    r = row(a.work, a.name, a.bench)
    json.dump(r, open(f"{a.work}/full_row_{a.name}.json", "w"), indent=1)
    ref = json.load(open(a.ref)).get(a.bench, {})
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"
    print("| 공간 | " + " | ".join(c[1] for c in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for n, v in [(f"**{a.name}**", r)] + list(ref.items()):
        print(f"| {n} | " + " | ".join(fmt(v.get(k)) for k, _ in COLS) + " |")
