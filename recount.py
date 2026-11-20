"""Перерахунок вердиктів з іншими порогами — без повторного аналізу фото.

    python recount.py --run 3                        # пороги, збережені для прогону
    python recount.py --run 3 --nomination quantity  # пороги пресету
    python recount.py --legacy-csv ~/Desktop/wle_results/progress.csv

Для старого progress.csv (формат v2) поріг упевненості детектора людей
і поріг дублікатів змінити неможливо: там збережено лише результат.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from wle_filter import config
from wle_filter.config import Thresholds
from wle_filter.store import Store
from wle_filter.verdict import evaluate, reasons_text


def evaluate_run(store: Store, run_id: int, thresholds: dict | None = None):
    run = store.get_run(run_id)
    t = Thresholds.from_dict(thresholds or run["thresholds"])
    images = store.images(run_id)
    return run, t, images, evaluate(images, t, store.pairs(run_id))


def print_summary(store: Store, run_id: int, thresholds: dict | None = None):
    run, t, images, ev = evaluate_run(store, run_id, thresholds)
    counts = {"ok": 0, "flagged": 0, "error": 0, "pending": 0}
    reasons: dict[str, int] = {}
    for e in ev:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
        for r in e["reasons"]:
            reasons[r["code"]] = reasons.get(r["code"], 0) + 1
    print("=" * 50)
    print(f"Прогін #{run_id}: {run['name']}")
    print("=" * 50)
    print(f"  Всього:   {len(ev)}")
    print(f"  OK:       {counts['ok']}")
    print(f"  Flagged:  {counts['flagged']}")
    print(f"  Errors:   {counts['error']}")
    if counts["pending"]:
        print(f"  В черзі:  {counts['pending']}")
    for k, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {k:10s} {n}")
    return counts


def export_csv(store: Store, run_id: int, thresholds: dict | None = None, path: str | None = None) -> str:
    run, t, images, ev = evaluate_run(store, run_id, thresholds)
    titles = {i["id"]: i["title"] for i in images}
    path = path or str(store.path.parent / f"run_{run_id}_results.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "page_id", "author", "blur_score", "person_area", "sign_coverage",
                    "horizon_angle", "horizon_conf", "shadows", "sat_clip", "text",
                    "duplicate_of", "verdict", "auto_verdict", "manual", "reasons", "local_path"])
        for img, e in zip(images, ev):
            dup = next((titles.get(r["of"], "") for r in e["reasons"] if r["code"] == "duplicate"), "")
            w.writerow([img["title"], img["page_id"] or "", img["author"] or "",
                        "" if img["blur"] is None else img["blur"], e["person_area"],
                        "" if img["sign"] is None else img["sign"],
                        *["" if img.get(k) is None else img[k] for k in
                          ("horizon_angle", "horizon_conf", "shadows", "sat_clip")],
                        img.get("text") or "", dup,
                        e["verdict"], e["auto_verdict"], e["manual"] or "",
                        img["error"] if img["status"] == "error" else reasons_text(e["reasons"]),
                        img["local_path"] or ""])
    return path


def recount_legacy_csv(path: str, t: Thresholds) -> tuple[list[dict], dict]:
    """Перерахунок старого progress.csv (формат v2 main.py)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = {"ok": 0, "flagged": 0, "error": 0}
    for r in rows:
        if r.get("verdict") == "error":
            counts["error"] += 1
            continue
        item = {
            "blur": float(r["blur_score"]) if r.get("blur_score") else None,
            "sign": float(r["sign_coverage"]) if r.get("sign_coverage") else None,
            # у v2 зберігалася лише площа найбільшої людини з conf ≥ 0.25
            "persons": [{"conf": 1.0, "area": float(r["person_area"])}] if r.get("person_area") else [],
        }
        from wle_filter.verdict import item_reasons
        reasons = item_reasons(item, t)
        if t.duplicate_enabled and r.get("duplicate_of"):
            reasons.append({"code": "duplicate", "value": "?"})
        r["reasons"] = reasons_text(reasons)
        r["verdict"] = "flagged" if reasons else "ok"
        counts[r["verdict"]] += 1
    return rows, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Перерахунок вердиктів WLE Filter")
    ap.add_argument("--run", type=int, help="номер прогону в базі")
    ap.add_argument("--nomination", help="взяти пороги з пресету (quality | quantity)")
    ap.add_argument("--legacy-csv", help="старий progress.csv")
    ap.add_argument("--out", help="куди записати CSV")
    args = ap.parse_args(argv)

    thresholds = None
    if args.nomination:
        presets = config.load_presets()
        if args.nomination not in presets:
            print(f"Невідома номінація: {args.nomination}")
            return 2
        thresholds = presets[args.nomination]["thresholds"]

    if args.legacy_csv:
        t = Thresholds.from_dict(thresholds)
        rows, counts = recount_legacy_csv(args.legacy_csv, t)
        out = args.out or os.path.join(os.path.dirname(args.legacy_csv), "results_recounted.csv")
        if rows:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        print(f"Всього: {len(rows)}  OK: {counts['ok']}  Flagged: {counts['flagged']}  Errors: {counts['error']}")
        print(f"CSV: {out}")
        return 0

    store = Store()
    run_id = args.run or (store.list_runs() or [{}])[0].get("id")
    if not run_id or not store.get_run(run_id):
        print("Прогін не знайдено. Вкажіть --run або створіть прогін: python main.py run …")
        return 1
    print_summary(store, run_id, thresholds)
    print(f"CSV: {export_csv(store, run_id, thresholds, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
