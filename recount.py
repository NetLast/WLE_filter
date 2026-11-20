import csv, os, html, time, requests

# ─── НАЛАШТУВАННЯ ─────────────────────────────────────────────
HASH_THRESHOLD        = 12
HASH_TYPE             = "dhash"
BLUR_THRESHOLD        = 200
PERSON_CONFIDENCE     = 0.25
PERSON_AREA_THRESHOLD = 0.05
SIGN_COVERAGE_MIN     = 0.15

INPUT_CSV  = os.path.join(os.path.expanduser("~"), "Desktop", "wle_results", "progress.csv")
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "wle_results")

HEADERS = {"User-Agent": "WLE-Filter-Bot/1.0 (research; commons.wikimedia.org)"}


# ─── Отримати URL з Wikimedia для списку файлів ───────────────
def get_thumb_urls_batch(file_titles):
    """Отримує thumb URL для до 50 файлів за один запит."""
    urls = {}
    titles_str = "|".join(file_titles)
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action":   "query",
                "titles":   titles_str,
                "prop":     "imageinfo",
                "iiprop":   "url",
                "iiurlwidth": 400,
                "format":   "json"
            },
            headers=HEADERS, timeout=30
        )
        data = r.json()
        for page in data["query"]["pages"].values():
            title = page.get("title", "")
            info  = page.get("imageinfo", [{}])[0]
            url   = info.get("thumburl") or info.get("url", "")
            if url:
                urls[title] = url
    except Exception as e:
        print(f"  Помилка API: {e}")
    return urls


def get_commons_url(file_title):
    """Повертає посилання на сторінку файлу на Commons."""
    name = file_title.replace("File:", "").replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/File:{name}"


# ─── Перерахунок ──────────────────────────────────────────────
def recount():
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    counts  = {"ok": 0, "flagged": 0, "error": 0}

    for r in rows:
        if r["verdict"] == "error":
            counts["error"] += 1
            r["reasons"] = r.get("reasons", "error")
            results.append(r)
            continue

        reasons = []

        blur = float(r["blur_score"]) if r["blur_score"] else 9999
        if blur < BLUR_THRESHOLD:
            reasons.append(f"blur:{blur}")

        person = float(r["person_area"]) if r["person_area"] else 0
        if person >= PERSON_AREA_THRESHOLD:
            reasons.append(f"person:{person:.1%}")

        sign = float(r["sign_coverage"]) if r["sign_coverage"] else 0
        if sign >= SIGN_COVERAGE_MIN:
            reasons.append(f"sign:{sign:.1%}")

        if r.get("duplicate_of"):
            reasons.append(f"duplicate:{r['duplicate_of'][:40]}")

        r["reasons"] = " | ".join(reasons)
        r["verdict"] = "flagged" if reasons else "ok"
        counts[r["verdict"]] += 1
        results.append(r)

    # ─── Отримати URL для відсіяних ───────────────────────────
    flagged = [r for r in results if r["verdict"] == "flagged"]
    ok      = [r for r in results if r["verdict"] == "ok"]
    errors  = [r for r in results if r["verdict"] == "error"]

    print(f"Отримую URL для {len(flagged)} відсіяних фото з Wikimedia...")
    thumb_urls = {}

    # Батчами по 50 файлів за раз
    batch_size = 50
    flagged_titles = [r["file"] for r in flagged]
    for i in range(0, len(flagged_titles), batch_size):
        batch = flagged_titles[i:i+batch_size]
        batch_urls = get_thumb_urls_batch(batch)
        thumb_urls.update(batch_urls)
        print(f"  {min(i+batch_size, len(flagged_titles))}/{len(flagged_titles)}...", end="\r")
        time.sleep(0.5)
    print(f"\nURL отримано: {len(thumb_urls)}")

    # ─── Зберегти CSV ─────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "results_recounted.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV: {out_csv}")

    # ─── HTML галерея з URL ───────────────────────────────────
    gallery_path = os.path.join(OUTPUT_DIR, "flagged_gallery_recounted.html")
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Flagged — WLE Ukraine 2026</title>
<style>
body{{font-family:Arial;margin:20px;background:#f4f4f4;}}
h1{{color:#333;font-size:clamp(16px,4vw,24px);}}
.stats{{background:#fff;padding:12px 20px;border-radius:8px;
        margin-bottom:20px;display:flex;gap:16px;flex-wrap:wrap;}}
.stat{{text-align:center;min-width:60px;}}
.stat b{{display:block;font-size:clamp(18px,5vw,22px);}}
.grid{{display:grid;
       grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));
       gap:16px;}}
.card{{background:#fff;border-radius:8px;padding:10px;
       box-shadow:0 2px 5px rgba(0,0,0,.1);}}
img{{width:100%;border-radius:5px;display:block;}}
.noimg{{width:100%;height:170px;background:#ddd;display:flex;
        align-items:center;justify-content:center;color:#777;
        border-radius:5px;font-size:13px;text-align:center;padding:10px;
        box-sizing:border-box;}}
.reason{{color:#c62828;font-size:13px;margin:6px 0;}}
.name{{font-size:11px;color:#555;word-break:break-all;margin:4px 0;}}
.commons-link{{font-size:12px;}}
</style></head><body>
<h1>Flagged Photos — Wiki Loves Earth Ukraine 2026</h1>
<div class="stats">
  <div class="stat"><b>{len(results)}</b>всього</div>
  <div class="stat"><b style="color:#2e7d32">{len(ok)}</b>пройшли</div>
  <div class="stat"><b style="color:#c62828">{len(flagged)}</b>відсіяно</div>
  <div class="stat"><b style="color:#888">{len(errors)}</b>помилки</div>
</div>
<div class="grid">
""")
        for r in flagged:
            title     = r["file"]
            safe_name = html.escape(title)
            reason    = html.escape(str(r.get("reasons", "")))
            thumb_url = thumb_urls.get(title, "")
            commons_url = get_commons_url(title)

            if thumb_url:
                img_tag = (f'<a href="{html.escape(commons_url)}" target="_blank">'
                           f'<img src="{html.escape(thumb_url)}" loading="lazy"></a>')
            else:
                img_tag = (f'<div class="noimg">'
                           f'<a href="{html.escape(commons_url)}" target="_blank">'
                           f'фото не знайдено — відкрити на Commons</a></div>')

            f.write(
                f'<div class="card">{img_tag}'
                f'<p class="reason"><b>{reason}</b></p>'
                f'<p class="name">{safe_name}</p>'
                f'<p class="commons-link">'
                f'<a href="{html.escape(commons_url)}" target="_blank">→ Commons</a>'
                f'</p></div>\n'
            )
        f.write("</div></body></html>")

    print(f"Галерея: {gallery_path}")
    print(f"\n{'='*50}")
    print(f"ЗАВЕРШЕНО")
    print(f"{'='*50}")
    print(f"  Всього:   {len(results)}")
    print(f"  OK:       {counts['ok']}")
    print(f"  Flagged:  {counts['flagged']}")
    print(f"  Errors:   {counts['error']}")
    print(f"\nГалерею можна поширити — фото підвантажуються з Wikimedia")


if __name__ == "__main__":
    recount()