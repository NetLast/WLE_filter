import os, re, html, time, requests, cv2, csv, numpy as np, shutil
import imagehash
from PIL import Image
from tqdm import tqdm

# ─── НАЛАШТУВАННЯ ────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(os.path.expanduser("~"), "Desktop", "wle_results")
IMAGES_DIR    = os.path.join(OUTPUT_DIR, "images")
FLAGGED_DIR   = os.path.join(OUTPUT_DIR, "flagged_images")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.csv")

WLE_CATEGORY = "Wiki Loves Earth Ukraine 2026 - Quality"

HASH_THRESHOLD        = 12
HASH_TYPE             = "dhash"
BLUR_THRESHOLD        = 200
PERSON_CONFIDENCE     = 0.25
PERSON_AREA_THRESHOLD = 0.05
SIGN_COVERAGE_MIN     = 0.15

HEADERS = {"User-Agent": "WLE-Filter-Bot/1.0 (research; commons.wikimedia.org)"}
SLEEP_S = 0.7


def ensure_dirs():
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(IMAGES_DIR,  exist_ok=True)
    os.makedirs(FLAGGED_DIR, exist_ok=True)


# ─── Прогрес ─────────────────────────────────────────────────────────────────
def load_progress():
    done = {}
    if not os.path.exists(PROGRESS_FILE):
        return done
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row["file"]] = row
    print(f"Знайдено прогрес: {len(done)} вже оброблених файлів")
    return done


def save_record(record, is_first):
    mode = "w" if is_first else "a"
    with open(PROGRESS_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if is_first:
            writer.writeheader()
        writer.writerow(record)


# ─── Wikimedia API ────────────────────────────────────────────────────────────
def get_category_files(category):
    files = []
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": "Category:" + category,
        "cmtype":  "file",
        "cmlimit": "500",
        "format":  "json"
    }
    print("Завантажую список файлів...")
    while True:
        r = requests.get("https://commons.wikimedia.org/w/api.php",
                         params=params, headers=HEADERS, timeout=30)
        data = r.json()
        batch = data["query"]["categorymembers"]
        files.extend([f["title"] for f in batch])
        print(f"  {len(files)} файлів...", end="\r")
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
        time.sleep(0.3)
    print(f"\nВсього файлів: {len(files)}")
    return files


def get_thumb_url(file_title):
    title = file_title if file_title.startswith("File:") else f"File:{file_title}"
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={"action": "query", "titles": title,
                    "prop": "imageinfo", "iiprop": "url",
                    "iiurlwidth": 800, "format": "json"},
            headers=HEADERS, timeout=15)
        if r.status_code == 200:
            for p in r.json().get("query", {}).get("pages", {}).values():
                ii = p.get("imageinfo", [{}])[0]
                return ii.get("thumburl") or ii.get("url", "")
    except (requests.RequestException, ValueError):
        pass
    return ""


def safe_filename(file_title):
    """Прибираємо префікс File: і замінюємо заборонені символи."""
    name = re.sub(r'^File:', '', file_title)
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_or_download_image(file_title):
    new_name = safe_filename(file_title)                          # без File:
    old_name = re.sub(r'[<>:"/\\|?*]', '_', file_title)         # старий формат з File_

    local_path_new = os.path.join(IMAGES_DIR, new_name)
    local_path_old = os.path.join(IMAGES_DIR, old_name)

    # Шукаємо спочатку старий формат (вже скачані раніше)
    if os.path.exists(local_path_old) and os.path.getsize(local_path_old) > 15000:
        return local_path_old

    # Потім новий
    if os.path.exists(local_path_new) and os.path.getsize(local_path_new) > 15000:
        return local_path_new

    # Нічого нема — качаємо
    thumb_url = get_thumb_url(file_title)
    if not thumb_url:
        return ""

    for attempt in range(8):
        try:
            r = requests.get(thumb_url, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    return ""
                with open(local_path_new, "wb") as f:
                    f.write(r.content)
                time.sleep(SLEEP_S)
                return local_path_new
            elif r.status_code == 429:
                wait = (2 ** attempt) * 3
                print(f"  429 — чекаємо {wait}с...")
                time.sleep(wait)
            elif 400 <= r.status_code < 500:
                return ""
            else:
                time.sleep(3)
        except requests.RequestException:
            time.sleep(3)
    return ""


# ─── Фільтри ─────────────────────────────────────────────────────────────────
def check_blur(img_array):
    h, w = img_array.shape[:2]
    if max(h, w) > 600:
        scale = 600 / max(h, w)
        img_array = cv2.resize(img_array, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD, round(score, 1)


def check_person(img_array, yolo_model):
    h, w   = img_array.shape[:2]
    res    = yolo_model(img_array, verbose=False)[0]
    max_fr = 0.0
    for box in res.boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) >= PERSON_CONFIDENCE:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            max_fr = max(max_fr, (x2-x1)*(y2-y1)/(h*w))
    return max_fr >= PERSON_AREA_THRESHOLD, round(max_fr, 3)


def check_sign_or_plaque(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(bw, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    text_area = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1.0)
        area   = cw * ch
        if 2 < aspect < 60 and 8 < ch < h * 0.2 and area > 200:
            text_area += area
    coverage = text_area / (h * w)
    return coverage > SIGN_COVERAGE_MIN, round(coverage, 3)


phash_registry = []


def check_duplicate(pil_img, file_title):
    h = imagehash.dhash(pil_img) if HASH_TYPE == "dhash" else imagehash.phash(pil_img)
    for prev_h, prev_title in phash_registry:
        dist = h - prev_h
        if dist <= HASH_THRESHOLD:
            return True, f"similar(dist={dist})", prev_title
    phash_registry.append((h, file_title))
    return False, None, None


def decide_verdict(record):
    if record.get("verdict") == "error":
        return "error"
    return "flagged" if record.get("reasons") else "ok"


# ─── HTML-галерея ─────────────────────────────────────────────────────────────
def create_flagged_gallery(results):
    output_path = os.path.join(OUTPUT_DIR, "flagged_gallery.html")
    flagged = [r for r in results if r.get("verdict") == "flagged"]
    ok      = [r for r in results if r.get("verdict") == "ok"]
    errors  = [r for r in results if r.get("verdict") == "error"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Flagged — WLE Ukraine 2026</title>
<style>
body{{font-family:Arial;margin:20px;background:#f4f4f4;}}
h1{{color:#333;}}
.stats{{background:#fff;padding:12px 20px;border-radius:8px;
        margin-bottom:20px;display:flex;gap:24px;}}
.stat{{text-align:center;}} .stat b{{display:block;font-size:22px;}}
.grid{{display:flex;flex-wrap:wrap;gap:16px;}}
.card{{background:#fff;border-radius:8px;padding:10px;width:320px;
       box-shadow:0 2px 5px rgba(0,0,0,.1);}}
img{{max-width:300px;border-radius:5px;display:block;}}
.noimg{{width:300px;height:170px;background:#ddd;display:flex;
        align-items:center;justify-content:center;color:#777;border-radius:5px;}}
.reason{{color:#c62828;font-size:13px;margin:6px 0;}}
.name{{font-size:12px;color:#555;word-break:break-all;}}
</style></head><body>
<h1>Flagged Photos — Wiki Loves Earth Ukraine 2026</h1>
<div class="stats">
  <div class="stat"><b>{len(results)}</b>всього</div>
  <div class="stat"><b style="color:#2e7d32">{len(ok)}</b>пройшли</div>
  <div class="stat"><b style="color:#c62828">{len(flagged)}</b>відсіяно</div>
  <div class="stat"><b style="color:#888">{len(errors)}</b>помилки</div>
</div><div class="grid">
""")
        for r in flagged:
            local_path = r.get("local_path")
            filename   = os.path.basename(local_path) if local_path else r.get("file", "")
            safe_name  = html.escape(str(filename))
            reason     = html.escape(str(r.get("reasons", "")))
            img_tag    = (f'<img src="images/{safe_name}" loading="lazy">'
                          if local_path
                          else '<div class="noimg">фото не завантажено</div>')
            f.write(
                f'<div class="card">{img_tag}'
                f'<p class="reason"><b>{reason}</b></p>'
                f'<p class="name">{safe_name}</p></div>\n'
            )
        f.write("</div></body></html>")
    print(f"HTML-галерея: {output_path}")


# ─── Основний цикл ────────────────────────────────────────────────────────────
def run_filter():
    from ultralytics import YOLO
    print("WLE Filter v2.1")
    ensure_dirs()
    phash_registry.clear()

    print("Завантаження YOLOv8n...")
    yolo_model = YOLO("yolov8n.pt")

    all_files = get_category_files(WLE_CATEGORY)
    done      = load_progress()
    todo      = [f for f in all_files if f not in done]
    print(f"Залишилось обробити: {len(todo)} з {len(all_files)}")

    is_first_record = len(done) == 0
    counts = {"ok": 0, "flagged": 0, "error": 0}
    for r in done.values():
        counts[r.get("verdict", "ok")] = counts.get(r.get("verdict", "ok"), 0) + 1

    for file_title in tqdm(todo, desc="Обробка"):
        record = {
            "file":          file_title,
            "blur_score":    None,
            "person_area":   None,
            "sign_coverage": None,
            "duplicate_of":  None,
            "local_path":    None,
            "verdict":       "ok",
            "reasons":       []
        }

        try:
            local_path = get_or_download_image(file_title)
            if not local_path:
                record["verdict"] = "error"
                record["reasons"].append("download_failed")
            else:
                record["local_path"] = local_path
                pil_img = Image.open(local_path).convert("RGB")
                np_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                is_dup, dup_type, dup_of = check_duplicate(pil_img, file_title)
                if is_dup:
                    record["duplicate_of"] = dup_of
                    record["reasons"].append(f"duplicate:{dup_type}")

                is_blurry, blur_score = check_blur(np_img)
                record["blur_score"] = blur_score
                if is_blurry:
                    record["reasons"].append(f"blur:{blur_score}")

                has_person, person_area = check_person(np_img, yolo_model)
                record["person_area"] = person_area
                if has_person:
                    record["reasons"].append(f"person:{person_area:.1%}")

                is_sign, sign_cov = check_sign_or_plaque(np_img)
                record["sign_coverage"] = sign_cov
                if is_sign:
                    record["reasons"].append(f"sign:{sign_cov:.1%}")

            record["verdict"] = decide_verdict(record)

        except Exception as e:
            record["verdict"] = "error"
            record["reasons"].append(f"error:{str(e)[:80]}")

        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
        record["reasons"] = " | ".join(record["reasons"])

        save_record(record, is_first=is_first_record)
        is_first_record = False

    # Фінальна галерея
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        all_results = list(csv.DictReader(f))

    create_flagged_gallery(all_results)

    print("\nКопіювання flagged фото...")
    for r in all_results:
        if r.get("verdict") != "ok" and r.get("local_path"):
            try:
                shutil.copy2(r["local_path"],
                             os.path.join(FLAGGED_DIR,
                                          os.path.basename(r["local_path"])))
            except OSError:
                pass

    print("\n" + "=" * 50)
    print("ЗАВЕРШЕНО")
    print("=" * 50)
    print(f"  Всього:   {len(all_results)}")
    print(f"  OK:       {counts['ok']}")
    print(f"  Flagged:  {counts['flagged']}")
    print(f"  Errors:   {counts['error']}")
    print(f"\nРезультати: {OUTPUT_DIR}")
    print(f"  flagged_gallery.html — відкрий у браузері")
    print(f"  progress.csv — всі результати")
    print(f"  flagged_images/ — відсіяні фото")


if __name__ == "__main__":
    run_filter()