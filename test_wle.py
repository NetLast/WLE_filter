"""Тести WLE Filter.  Запуск:  python -m pytest -q   або   python test_wle.py"""
import csv
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Ізолюємо дані тестів від робочої бази
_TMP = tempfile.mkdtemp(prefix="wle_test_")
os.environ["WLE_DATA_DIR"] = _TMP

from wle_filter import analysis, config, sources  # noqa: E402
from wle_filter.config import Thresholds  # noqa: E402
from wle_filter.store import Store  # noqa: E402
from wle_filter.verdict import evaluate, find_duplicates, person_area, reasons_text  # noqa: E402
from wle_filter.worker import Worker  # noqa: E402
import recount  # noqa: E402


# ─── Хелпери ─────────────────────────────────────────────────────────────────
def sharp_img(size=300):
    ch = (np.indices((size, size)).sum(axis=0) // 20 % 2 * 255).astype(np.uint8)
    return np.stack([ch] * 3, axis=-1)


def blurry_img(size=300):
    return cv2.GaussianBlur(sharp_img(size), (51, 51), 20)


def blank_img(size=300, val=200):
    return np.full((size, size, 3), val, dtype=np.uint8)


def to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def noise_pil(seed, size=256):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.NEAREST)


def fake_yolo(boxes):
    """boxes: [(cls, conf, [x1,y1,x2,y2])]"""
    fb = []
    for cls, conf, xyxy in boxes:
        b = MagicMock()
        b.cls, b.conf, b.xyxy = [cls], [conf], [np.array(xyxy, dtype=float)]
        fb.append(b)
    res = MagicMock()
    res.boxes = fb
    model = MagicMock(return_value=[res])
    return model


def item(i, **kw):
    base = {"id": i, "status": "done", "blur": 5000, "sign": 0.0, "persons": [],
            "dhash": None, "phash": None, "width": 4000, "height": 3000, "author": "A", "manual": None}
    base.update(kw)
    return base


# ═══ config ══════════════════════════════════════════════════════════════════
class TestThresholds(unittest.TestCase):
    def test_from_dict_coerces_types(self):
        t = Thresholds.from_dict({"blur_threshold": "150", "duplicate_distance": "7.0",
                                  "blur_enabled": "false", "unknown": 1})
        self.assertEqual(t.blur_threshold, 150.0)
        self.assertEqual(t.duplicate_distance, 7)
        self.assertFalse(t.blur_enabled)

    def test_invalid_enums_fallback(self):
        t = Thresholds.from_dict({"hash_type": "md5", "duplicate_scope": "x"})
        self.assertEqual((t.hash_type, t.duplicate_scope), ("dhash", "all"))

    def test_confidence_not_below_floor(self):
        self.assertEqual(Thresholds.from_dict({"person_confidence": 0}).person_confidence, 0.25)

    def test_presets_have_both_nominations(self):
        p = config.load_presets()
        self.assertIn("quality", p)
        self.assertIn("quantity", p)

    def test_defaults_match_original_constants(self):
        t = Thresholds()
        self.assertEqual((t.blur_threshold, t.person_confidence, t.person_area, t.sign_coverage,
                          t.duplicate_distance, t.hash_type),
                         (200, 0.25, 0.05, 0.15, 12, "dhash"))

    def test_save_and_reset_preset(self):
        config.save_preset("quality", {**Thresholds().to_dict(), "blur_threshold": 333})
        self.assertEqual(config.load_presets()["quality"]["thresholds"]["blur_threshold"], 333)
        config.reset_preset("quality")
        self.assertEqual(config.load_presets()["quality"]["thresholds"]["blur_threshold"], 200)


# ═══ analysis ════════════════════════════════════════════════════════════════
class TestAnalysis(unittest.TestCase):
    def test_sharp_vs_blurry(self):
        self.assertGreater(analysis.blur_score(sharp_img()), 200)
        self.assertLess(analysis.blur_score(blurry_img()), 200)

    def test_blank_is_zero_blur(self):
        self.assertEqual(analysis.blur_score(blank_img()), 0.0)

    def test_large_image_ok(self):
        self.assertIsInstance(analysis.blur_score(sharp_img(1200)), float)

    def test_person_detections_normalized(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        dets = analysis.person_detections(img, fake_yolo([(0, 0.9, [0, 0, 100, 50]), (2, 0.99, [0, 0, 200, 100])]))
        self.assertEqual(len(dets), 1)                          # машина відкинута
        self.assertAlmostEqual(dets[0]["area"], 0.25, places=3)
        self.assertEqual(dets[0]["box"], [0.0, 0.0, 0.5, 0.5])

    def test_person_detections_yolo_defaults(self):
        # як у вихідному check_person: модель викликається без зміни conf
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        model = fake_yolo([(0, 0.3, [0, 0, 50, 50])])
        analysis.person_detections(img, model)
        self.assertNotIn("conf", model.call_args.kwargs)
        self.assertNotIn("classes", model.call_args.kwargs)

    def test_sign_blank_zero(self):
        self.assertEqual(analysis.sign_coverage(blank_img()), 0.0)

    def test_sign_in_range(self):
        cov = analysis.sign_coverage(sharp_img())
        self.assertTrue(0.0 <= cov <= 1.0)

    def test_load_for_analysis_resizes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "big.jpg")
            Image.new("RGB", (3000, 2000), (100, 120, 90)).save(p)
            im = analysis.load_for_analysis(p)
            self.assertEqual(im.size, (config.ANALYSIS_WIDTH, round(2000 * config.ANALYSIS_WIDTH / 3000)))

    def test_analyze_keys(self):
        m = analysis.analyze(to_pil(sharp_img()), fake_yolo([]))
        self.assertEqual(set(m), {"blur", "persons", "sign", "dhash", "phash", "text_area", "text",
                                  "watermark", "text_boxes", "sign_object", "sign_object_box"})
        self.assertEqual(len(m["dhash"]), 16)


# ═══ verdict ═════════════════════════════════════════════════════════════════
class TestVerdict(unittest.TestCase):
    def test_ok_when_clean(self):
        self.assertEqual(evaluate([item(1)], Thresholds())[0]["verdict"], "ok")

    def test_blur_threshold_live(self):
        it = [item(1, blur=150)]
        self.assertEqual(evaluate(it, Thresholds(blur_threshold=200))[0]["verdict"], "flagged")
        self.assertEqual(evaluate(it, Thresholds(blur_threshold=100))[0]["verdict"], "ok")
        self.assertEqual(evaluate(it, Thresholds(blur_enabled=False))[0]["verdict"], "ok")

    def test_person_confidence_live(self):
        persons = [{"conf": 0.3, "area": 0.2}, {"conf": 0.9, "area": 0.02}]
        self.assertEqual(person_area(persons, 0.25), 0.2)
        self.assertEqual(person_area(persons, 0.5), 0.02)
        it = [item(1, persons=persons)]
        self.assertEqual(evaluate(it, Thresholds(person_confidence=0.25))[0]["verdict"], "flagged")
        self.assertEqual(evaluate(it, Thresholds(person_confidence=0.5))[0]["verdict"], "ok")

    def test_sign(self):
        it = [item(1, sign=0.3)]
        r = evaluate(it, Thresholds(sign_coverage=0.15))[0]
        self.assertEqual(r["reasons"][0]["code"], "sign")
        # як у вихідному коді: coverage > SIGN_COVERAGE_MIN (строго)
        self.assertEqual(evaluate([item(1, sign=0.15)], Thresholds(sign_coverage=0.15))[0]["verdict"], "ok")

    def test_duplicate_first_match_not_closest(self):
        its = [item(1, dhash="000000000000000f"), item(2, dhash="0000000000000ff0"),
               item(3, dhash="00000000000000ff")]
        d = find_duplicates(its, Thresholds(duplicate_distance=8))
        self.assertEqual(d[3], (1, 4))   # перший у реєстрі, хоча 2 теж у межах

    def test_min_mpx(self):
        it = [item(1, width=1000, height=1000)]
        self.assertEqual(evaluate(it, Thresholds())[0]["verdict"], "ok")
        r = evaluate(it, Thresholds(min_mpx_enabled=True, min_mpx=2))[0]
        self.assertEqual(r["reasons"][0], {"code": "mpx", "value": 1.0})

    def test_error_and_pending(self):
        res = evaluate([item(1, status="error", error="download_failed"), item(2, status="pending")], Thresholds())
        self.assertEqual([r["verdict"] for r in res], ["error", "pending"])

    def test_manual_overrides(self):
        res = evaluate([item(1, blur=1, manual="accept"), item(2, manual="reject")], Thresholds())
        self.assertEqual([r["verdict"] for r in res], ["ok", "flagged"])
        self.assertEqual([r["auto_verdict"] for r in res], ["flagged", "ok"])

    def test_duplicates_first_is_original(self):
        h = "ffff0000ffff0000"
        res = evaluate([item(1, dhash=h), item(2, dhash="ffff0000ffff0001")], Thresholds())
        self.assertEqual(res[0]["verdict"], "ok")
        self.assertEqual(res[1]["reasons"][0], {"code": "duplicate", "value": 1, "of": 1})

    def test_duplicate_distance_live(self):
        its = [item(1, dhash="0000000000000000"), item(2, dhash="00000000000000ff")]  # dist 8
        self.assertIn(2, find_duplicates(its, Thresholds(duplicate_distance=8)))
        self.assertNotIn(2, find_duplicates(its, Thresholds(duplicate_distance=7)))

    def test_duplicate_scope_author(self):
        its = [item(1, dhash="00" * 8, author="A"), item(2, dhash="00" * 8, author="B")]
        self.assertIn(2, find_duplicates(its, Thresholds(duplicate_scope="all")))
        self.assertEqual(find_duplicates(its, Thresholds(duplicate_scope="author")), {})

    def test_duplicate_not_registered(self):
        # 2 — дублікат 1; 3 близьке до 2, але далеке від 1 → не дублікат (як у v2)
        its = [item(1, dhash="0000000000000000"), item(2, dhash="000000000000000f"),
               item(3, dhash="00000000000000ff")]
        d = find_duplicates(its, Thresholds(duplicate_distance=4))
        self.assertEqual(set(d), {2})

    def test_phash_used_when_selected(self):
        its = [item(1, dhash="00" * 8, phash="ff" * 8), item(2, dhash="ff" * 8, phash="ff" * 8)]
        self.assertEqual(find_duplicates(its, Thresholds(hash_type="dhash")), {})
        self.assertIn(2, find_duplicates(its, Thresholds(hash_type="phash")))

    def test_real_hashes_brightness_shift(self):
        a = to_pil(sharp_img())
        b = ImageEnhance.Brightness(a).enhance(1.1)
        hs = [analysis.image_hashes(x)["dhash"] for x in (a, b)]
        self.assertIn(2, find_duplicates([item(1, dhash=hs[0]), item(2, dhash=hs[1])], Thresholds()))

    def test_reasons_text_legacy_format(self):
        s = reasons_text([{"code": "blur", "value": 12.5}, {"code": "person", "value": 0.12},
                          {"code": "duplicate", "value": 3, "of": 1}])
        self.assertEqual(s, "blur:12.5 | person:12.0% | duplicate:similar(dist=3)")


# ═══ sources ═════════════════════════════════════════════════════════════════
class TestSources(unittest.TestCase):
    def test_commons_page_url(self):
        self.assertEqual(sources.commons_page_url("File:my photo.jpg"),
                         "https://commons.wikimedia.org/wiki/File:my_photo.jpg")

    def test_local_folder_authors_and_dims(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/Author1")
            Image.new("RGB", (40, 30)).save(f"{d}/Author1/a.jpg")
            Image.new("RGB", (20, 10)).save(f"{d}/b.png")
            Path(f"{d}/notes.txt").write_text("x")
            c = {x["key"]: x for x in sources.list_local_folder(d)}
            self.assertEqual(set(c), {"Author1/a.jpg", "b.png"})
            self.assertEqual(c["Author1/a.jpg"]["author"], "Author1")
            self.assertIsNone(c["b.png"]["author"])
            self.assertEqual((c["Author1/a.jpg"]["width"], c["Author1/a.jpg"]["height"]), (40, 30))

    def test_local_folder_missing(self):
        with self.assertRaises(FileNotFoundError):
            sources.list_local_folder("/nonexistent/xyz")

    def test_commons_category_listing(self):
        def resp(data):
            r = MagicMock(status_code=200)
            r.json.return_value = data
            return r
        calls = [
            resp({"query": {"categorymembers": [{"pageid": 1, "title": "File:A.jpg", "timestamp": "2026-05-01T00:00:00Z"}]},
                  "continue": {"cmcontinue": "x", "continue": "-||"}}),
            resp({"query": {"categorymembers": [{"pageid": 2, "title": "File:B.jpg", "timestamp": "2026-05-02T00:00:00Z"}]}}),
            resp({"query": {"pages": [
                {"pageid": 1, "imageinfo": [{"user": "U1", "width": 4000, "height": 3000, "mime": "image/jpeg",
                                             "thumburl": "https://t/a.jpg", "timestamp": "2026-05-01T00:00:00Z"}]},
                {"pageid": 2, "imageinfo": [{"user": "U2", "width": 800, "height": 600, "mime": "image/jpeg",
                                             "thumburl": "https://t/b.jpg"}]}]}}),
        ]
        session = MagicMock()
        session.get.side_effect = calls
        with patch("wle_filter.sources.time.sleep"):
            out = sources.list_commons_category("Category:Test", session=session)
        self.assertEqual([o["page_id"] for o in out], [1, 2])
        self.assertEqual(out[0]["author"], "U1")
        self.assertEqual(out[1]["thumb_url"], "https://t/b.jpg")
        # друга сторінка запитана з cmcontinue
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["cmcontinue"], "x")


# ═══ store + worker ══════════════════════════════════════════════════════════
def make_folder(d):
    os.makedirs(f"{d}/A")
    os.makedirs(f"{d}/B")
    noise_pil(1).save(f"{d}/A/1.jpg")
    noise_pil(1).save(f"{d}/A/1_copy.jpg")      # дублікат у того ж автора
    noise_pil(1).save(f"{d}/B/1_other.jpg")     # такий самий кадр в іншого автора
    to_pil(blurry_img()).save(f"{d}/B/blur.jpg")
    noise_pil(2).save(f"{d}/B/2.jpg")


class TestStoreWorker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "db.sqlite3")
        self.photos = Path(self.tmp.name) / "photos"
        make_folder(self.photos)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_folder(self, nomination="quality"):
        t = config.load_presets()[nomination]["thresholds"]
        run_id = self.store.create_run("t", nomination, "folder", str(self.photos), t)
        w = Worker(self.store, yolo_loader=lambda: fake_yolo([]), ocr_loader=None)
        w.start(run_id)
        w.join(run_id, 30)
        return run_id

    def test_run_processes_all(self):
        run_id = self.run_folder()
        run = self.store.get_run(run_id)
        self.assertEqual(run["status"], "done")
        self.assertEqual((run["total"], run["processed"], run["errors"]), (5, 5, 0))
        self.assertTrue(all((self.store.cache_dir / str(run_id) / f"{i['id']}.jpg").exists()
                            for i in self.store.images(run_id)))

    def test_add_candidates_idempotent(self):
        run_id = self.store.create_run("t", "quality", "folder", str(self.photos), {})
        c = sources.list_local_folder(str(self.photos))
        self.assertEqual(self.store.add_candidates(run_id, c), 5)
        self.assertEqual(self.store.add_candidates(run_id, c), 0)

    def test_duplicate_scope_author_on_folder(self):
        run_id = self.run_folder()
        imgs = self.store.images(run_id)
        dup = lambda t: {e["id"] for e in evaluate(imgs, t) if any(r["code"] == "duplicate" for r in e["reasons"])}
        self.assertEqual(len(dup(Thresholds())), 2)                       # серед усіх
        self.assertEqual(len(dup(Thresholds(duplicate_scope="author"))), 1)

    def test_reanalyze_outdated(self):
        run_id = self.run_folder()
        self.assertEqual(self.store.reanalyze_all(run_id), 5)
        run = self.store.get_run(run_id)
        self.assertEqual((run["outdated"], run["processed"]), (5, 0))
        w = Worker(self.store, yolo_loader=lambda: fake_yolo([]), ocr_loader=None)
        w.start(run_id)
        w.join(run_id, 30)
        self.assertEqual(self.store.get_run(run_id)["processed"], 5)

    def test_migration_from_v31_db(self):
        import sqlite3
        p = Path(self.tmp.name) / "old.sqlite3"
        con = sqlite3.connect(p)
        con.executescript("CREATE TABLE images (id INTEGER PRIMARY KEY, run_id INTEGER, seq INTEGER, key TEXT,"
                          " title TEXT, status TEXT DEFAULT 'done', persons TEXT, mver INTEGER DEFAULT 2);"
                          "INSERT INTO images(run_id, seq, key, title) VALUES (1, 0, 'a', 'a');")
        con.close()
        st = Store(p)
        self.assertTrue(st.images(1)[0]["outdated"])
        st.close()

    def test_error_recorded_and_retry(self):
        Path(self.photos / "A" / "broken.jpg").write_bytes(b"not an image")
        run_id = self.run_folder()
        run = self.store.get_run(run_id)
        self.assertEqual(run["errors"], 1)
        self.assertEqual(self.store.retry_errors(run_id), 1)
        self.assertEqual(len(self.store.pending_images(run_id)), 1)

    def test_pause_and_resume(self):
        t = Thresholds().to_dict()
        run_id = self.store.create_run("t", "quality", "folder", str(self.photos), t)
        slow = fake_yolo([])
        orig = slow.side_effect

        def slow_call(*a, **k):
            time.sleep(0.2)
            return slow.return_value
        slow.side_effect = slow_call
        w = Worker(self.store, yolo_loader=lambda: slow, ocr_loader=None)
        w.start(run_id)
        time.sleep(0.3)
        w.pause(run_id)
        w.join(run_id, 10)
        self.assertEqual(self.store.get_run(run_id)["status"], "paused")
        self.assertLess(self.store.get_run(run_id)["processed"], 5)
        slow.side_effect = orig
        w.start(run_id)
        w.join(run_id, 10)
        self.assertEqual(self.store.get_run(run_id)["processed"], 5)

    def test_manual_decision(self):
        run_id = self.run_folder()
        i = self.store.images(run_id)[0]["id"]
        self.store.set_manual(i, "reject", "нотатка")
        img = self.store.image(i)
        self.assertEqual((img["manual"], img["note"]), ("reject", "нотатка"))
        self.store.set_manual(i, None)
        self.assertIsNone(self.store.image(i)["manual"])

    def test_recount_export_csv(self):
        run_id = self.run_folder()
        path = recount.export_csv(self.store, run_id)
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertTrue(any(r["reasons"].startswith("blur:") for r in rows))


class TestLegacyRecount(unittest.TestCase):
    def test_legacy_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "progress.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "blur_score", "person_area", "sign_coverage", "duplicate_of",
                            "local_path", "verdict", "reasons"])
                w.writerow(["File:a.jpg", "50", "0.0", "0.0", "", "", "flagged", "blur:50"])
                w.writerow(["File:b.jpg", "900", "0.0", "0.0", "", "", "ok", ""])
                w.writerow(["File:c.jpg", "", "", "", "", "", "error", "download_failed"])
            rows, counts = recount.recount_legacy_csv(p, Thresholds(blur_threshold=100))
            self.assertEqual(counts, {"ok": 1, "flagged": 1, "error": 1})
            rows, counts = recount.recount_legacy_csv(p, Thresholds(blur_threshold=40))
            self.assertEqual(counts["flagged"], 0)


# ═══ Таблички через OCR і вотермарки ═══════════════════════════════════════
def fake_ocr(lines):
    """lines: [(x0, y0, x1, y1, text, score)] у пікселях."""
    return MagicMock(return_value=([[[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], t, sc]
                                     for x0, y0, x1, y1, t, sc in lines], None))


class TestOcrSigns(unittest.TestCase):
    img = np.full((400, 800, 3), 120, np.uint8)

    def test_plaque_lines_grouped(self):
        r = analysis.text_signs(self.img, fake_ocr([
            (320, 200, 420, 215, "Дуб звичайний", 0.8),
            (330, 218, 410, 232, "Quercus robur", 0.82),
            (335, 236, 405, 250, "1400 років", 0.8)]))
        self.assertEqual([b[4] for b in r["text_boxes"]], ["sign"])     # одна табличка
        self.assertGreater(r["text_area"], 100 * 50 / (800 * 400))
        self.assertEqual(r["watermark"], "")
        self.assertIn("Quercus", r["text"])

    def test_short_word_with_repeat_is_text(self):
        # «АНАЛІЗИ», прочитане як «AHA», — справжній текст
        r = analysis.text_signs(self.img, fake_ocr([(300, 100, 500, 140, "AHA", 0.64)]))
        self.assertGreater(r["text_area"], 0)

    def test_watermark_near_edge(self):
        r = analysis.text_signs(self.img, fake_ocr([(10, 300, 110, 310, "OlegKanewsky", 0.75)]))
        self.assertEqual(r["watermark"], "OlegKanewsky")
        self.assertEqual(r["text_area"], 0.0)
        self.assertEqual(r["text_boxes"][0][4], "watermark")

    def test_junk_text_ignored(self):
        r = analysis.text_signs(self.img, fake_ocr([
            (300, 100, 400, 120, "1111", 0.9),       # візерунок
            (300, 150, 400, 170, "II", 0.9),         # візерунок
            (300, 200, 400, 220, "JUUU", 0.9),       # візерунок
            (300, 300, 400, 320, "A", 0.95),         # замало символів
            (300, 350, 400, 370, "", 0.0),           # текстура: порожній рядок
            (100, 100, 300, 140, "трава", 0.2)]))    # низька впевненість
        self.assertEqual((r["text_area"], r["watermark"], r["text_boxes"]), (0.0, "", []))

    def test_object_area_measured(self):
        img = np.full((400, 800, 3), (60, 140, 60), np.uint8)       # «трава»
        img[120:380, 300:500] = (150, 150, 150)                       # «камінь»
        img[200:250, 330:470] = (30, 30, 30)                          # «табличка»
        r = analysis.text_signs(img, fake_ocr([(340, 205, 460, 245, "Дуб звичайний", 0.8)]))
        self.assertAlmostEqual(r["sign_object"], 200 * 260 / (800 * 400), delta=0.03)

    def test_verdict_rules(self):
        t = Thresholds()
        ok = dict(text_area=0.005, sign_object=0.05, sign=0.0)
        self.assertEqual(evaluate([item(1, **ok)], t)[0]["verdict"], "ok")
        r = evaluate([item(1, text_area=0.03, sign_object=0.05)], t)[0]["reasons"]
        self.assertEqual((r[0]["code"], r[0]["how"]), ("sign", "ocr"))
        r = evaluate([item(1, text_area=0.005, sign_object=0.2)], t)[0]["reasons"]
        self.assertEqual((r[0]["code"], r[0]["how"]), ("sign", "object"))
        r = evaluate([item(1, text_area=0.0, watermark="Oleg")], t)[0]["reasons"]
        self.assertEqual(r, [{"code": "watermark", "value": "Oleg"}])
        self.assertEqual(evaluate([item(1, watermark="Oleg")], Thresholds(watermark_enabled=False))[0]["verdict"], "ok")

    def test_legacy_method_and_no_ocr_fallback(self):
        # метод «евристика» ігнорує OCR
        it = item(1, text_area=0.5, sign=0.0)
        self.assertEqual(evaluate([it], Thresholds(sign_method="legacy"))[0]["verdict"], "ok")
        # OCR не встановлено (text_area=None) — працює вихідна евристика
        it = item(1, text_area=None, sign=0.3)
        self.assertEqual(evaluate([it], Thresholds())[0]["reasons"][0]["how"], "legacy")

    def test_analyze_without_ocr_keeps_original_metrics(self):
        m = analysis.analyze(to_pil(sharp_img()), fake_yolo([]), None)
        self.assertIsNone(m["text_area"])
        self.assertEqual(m["sign"], analysis.sign_coverage(sharp_img()))


# ═══ Еквівалентність з вихідним main.py (перший коміт) ═══════════════════
ORIGINAL = Path(__file__).parent / "tools" / "original_main.py"


@unittest.skipUnless(ORIGINAL.exists(), "немає tools/original_main.py")
class TestMatchesOriginal(unittest.TestCase):
    """Метрики й вердикти нового коду мають збігатися з функціями першого коміту."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("original_main", ORIGINAL)
        cls.orig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.orig)
        rng = np.random.default_rng(7)
        cls.images = [sharp_img(400), blurry_img(400), blank_img(300), noise_pil(5, 512),
                      Image.fromarray(rng.integers(0, 255, (300, 500, 3), dtype=np.uint8))]

    def bgr(self, im):
        return im if isinstance(im, np.ndarray) else cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

    def test_blur_and_sign_identical(self):
        for im in self.images:
            b = self.bgr(im)
            _, score = self.orig.check_blur(b)
            self.assertEqual(analysis.blur_score(b), score)
            _, cov = self.orig.check_sign_or_plaque(b)
            self.assertEqual(analysis.sign_coverage(b), cov)
            for thr in (0.0, 0.05, 0.15, cov):
                self.orig.SIGN_COVERAGE_MIN = thr
                flagged, _ = self.orig.check_sign_or_plaque(b)
                ours = evaluate([item(1, sign=analysis.sign_coverage(b))],
                                Thresholds(sign_coverage=thr))[0]["verdict"] == "flagged"
                self.assertEqual(flagged, ours)

    def test_person_identical(self):
        model = fake_yolo([(0, 0.9, [0, 0, 60, 80]), (0, 0.3, [0, 0, 200, 200]), (2, 0.99, [0, 0, 300, 300])])
        b = np.zeros((300, 300, 3), np.uint8)
        for conf, area in ((0.25, 0.05), (0.5, 0.05), (0.25, 0.5), (0.95, 0.01)):
            self.orig.PERSON_CONFIDENCE, self.orig.PERSON_AREA_THRESHOLD = conf, area
            flagged, fr = self.orig.check_person(b, model)
            dets = analysis.person_detections(b, model)
            r = evaluate([item(1, persons=dets)], Thresholds(person_confidence=conf, person_area=area))[0]
            self.assertEqual(flagged, r["verdict"] == "flagged")
            self.assertEqual(round(fr, 3), r["person_area"])

    def test_duplicates_identical(self):
        pils = [noise_pil(i % 4, 256) for i in range(10)]
        pils[5] = ImageEnhance.Brightness(pils[1]).enhance(1.1)
        for thr in (0, 6, 12, 20, 30):
            self.orig.phash_registry.clear()
            self.orig.HASH_THRESHOLD, self.orig.HASH_TYPE = thr, "dhash"
            want = {}
            for i, p in enumerate(pils):
                is_dup, _, of = self.orig.check_duplicate(p, i)
                if is_dup:
                    want[i] = of
            its = [item(i, dhash=str(analysis.image_hashes(p)["dhash"])) for i, p in enumerate(pils)]
            got = {k: v[0] for k, v in find_duplicates(its, Thresholds(duplicate_distance=thr)).items()}
            self.assertEqual(got, want, f"threshold {thr}")


# ═══ API ═════════════════════════════════════════════════════════════════════
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from wle_filter.server import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        cls.photos = Path(cls.tmp.name) / "photos"
        make_folder(cls.photos)
        cls.store = Store(Path(cls.tmp.name) / "db.sqlite3")
        cls.worker = Worker(cls.store, yolo_loader=lambda: fake_yolo([]), ocr_loader=None)
        cls.client = TestClient(create_app(cls.store, cls.worker))

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tmp.cleanup()

    def create(self, **kw):
        body = {"nomination": "quality", "source_type": "folder", "source": str(self.photos), **kw}
        r = self.client.post("/api/runs", json=body)
        self.assertEqual(r.status_code, 201, r.text)
        rid = r.json()["id"]
        self.worker.join(rid, 30)
        return rid

    def results(self, rid, **kw):
        r = self.client.post(f"/api/runs/{rid}/results", json=kw)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_index_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("WLE Filter", r.text)

    def test_codex_assets_served(self):
        for f in ("codex.style.css", "codex-theme.css"):
            r = self.client.get(f"/static/codex/{f}")
            self.assertEqual(r.status_code, 200)
        self.assertIn('html[data-theme="dark"]', self.client.get("/static/codex/codex-theme.css").text)

    def test_presets(self):
        p = self.client.get("/api/presets").json()
        self.assertEqual(set(p) >= {"quality", "quantity"}, True)

    def test_bad_folder(self):
        r = self.client.post("/api/runs", json={"source_type": "folder", "source": "/nope/x"})
        self.assertEqual(r.status_code, 400)

    def test_full_flow(self):
        rid = self.create()
        res = self.results(rid, view="all")
        self.assertEqual(res["summary"]["total"], 5)
        self.assertEqual(res["summary"]["reasons"]["blur"], 1)
        self.assertEqual(res["summary"]["reasons"]["duplicate"], 2)

        # живі пороги: вимкнули дублікати — відсіяне лише розмите
        t = {**res["thresholds"], "duplicate_enabled": False}
        res2 = self.results(rid, view="flagged", thresholds=t)
        self.assertEqual(res2["matched"], 1)
        self.assertEqual(res2["items"][0]["reasons"][0]["code"], "blur")

        # ручне рішення
        iid = res2["items"][0]["id"]
        r = self.client.post(f"/api/images/{iid}/decision", json={"decision": "accept", "note": "ок"})
        self.assertEqual(r.json()["manual"], "accept")
        res3 = self.results(rid, view="flagged", thresholds=t)
        self.assertEqual(res3["matched"], 0)
        self.assertEqual(self.results(rid, view="manual")["matched"], 1)

        # збереження порогів і експорт
        self.client.put(f"/api/runs/{rid}/thresholds", json=t)
        self.assertFalse(self.client.get(f"/api/runs/{rid}").json()["thresholds"]["duplicate_enabled"])
        csv_text = self.client.get(f"/api/runs/{rid}/export?format=csv").text
        self.assertIn("auto_verdict", csv_text)
        js = self.client.get(f"/api/runs/{rid}/export?format=json").json()
        self.assertEqual(len(js["images"]), 5)
        self.assertTrue(all(i["verdict"] == "ok" for i in js["images"]))

        # мініатюра
        self.assertEqual(self.client.get(f"/api/images/{iid}/thumb").status_code, 200)

    def test_filters_reason_and_search(self):
        rid = self.create()
        res = self.results(rid, view="flagged", reason="duplicate")
        self.assertEqual(res["matched"], 2)
        self.assertEqual(self.results(rid, view="all", search="blur")["matched"], 1)
        self.assertEqual(self.results(rid, view="all", sort="blur")["items"][0]["title"], "blur.jpg")

    def test_bulk_and_delete(self):
        rid = self.create()
        ids = self.results(rid, view="flagged")["ids"]
        self.assertEqual(self.client.post(f"/api/runs/{rid}/decisions",
                                          json={"ids": ids, "decision": "reject"}).json()["updated"], len(ids))
        self.assertEqual(self.client.delete(f"/api/runs/{rid}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/runs/{rid}").status_code, 404)

    def test_fs_browse(self):
        r = self.client.get("/api/fs", params={"path": str(self.photos)}).json()
        self.assertEqual(sorted(r["dirs"]), ["A", "B"])


class TestCampaigns(unittest.TestCase):
    """Режими WLE / WLM: CSV-джерело, чорний список, роки, дублі в межах пам'ятки."""

    CSV = ("title,monument_id,categories\n"
           "1.jpg,63-101-0001,Cat A\n"
           "1_copy.jpg,63-101-0001,Cat A|Mass grave\n"
           "1_other.jpg,63-101-0002,Cat B\n"
           "blur.jpg,12-001-0003,Cat C\n"
           "2.jpg,63-101-0004,Cat D\n")

    def test_parse_csv_oblast_and_dedupe(self):
        rows = sources.parse_wlm_csv(self.CSV, "63")
        self.assertEqual([r["title"] for r in rows],
                         ["File:1.jpg", "File:1 copy.jpg", "File:1 other.jpg", "File:2.jpg"])
        self.assertEqual(len(sources.parse_wlm_csv(self.CSV, "")), 5)
        merged = sources.parse_wlm_csv(self.CSV + "1.jpg,63-101-0001,Cat Z\n", "63")
        self.assertEqual(merged[0]["categories"], "Cat A|Cat Z")
        self.assertEqual(len(merged), 4)

    def test_parse_csv_delimiters_and_errors(self):
        semi = self.CSV.replace(",", ";")
        self.assertEqual(len(sources.parse_wlm_csv("\ufeff" + semi, "63")), 4)
        with self.assertRaises(ValueError) as cm:
            sources.parse_wlm_csv("name,id\na,b\n")
        self.assertIn("title", str(cm.exception))

    def test_wlm_preset_matches_script(self):
        p = config.load_presets()["wlm"]
        t = Thresholds.from_dict(p["thresholds"])
        self.assertEqual((t.blur_threshold, t.person_confidence, t.person_area), (60.0, 0.5, 0.25))
        self.assertEqual((t.sign_method, t.sign_coverage, t.duplicate_distance), ("legacy", 0.45, 16))
        self.assertEqual((t.year_cutoff, t.duplicate_scope), (1954, "monument"))
        self.assertTrue(t.blacklist_enabled and t.year_enabled)
        # режими не змішуються: у WLE немає WLM-фільтрів і пресет іншої кампанії
        for k in ("quality", "quantity"):
            self.assertEqual(config.load_presets()[k]["campaign"], "wle")
            self.assertFalse(config.load_presets()[k]["thresholds"]["year_enabled"])

    def test_blacklist_year_and_monument_scope(self):
        t = Thresholds(blacklist_enabled=True, blacklist_categories="world war II graves\n\nMass Grave",
                       year_enabled=True, year_cutoff=1954)
        items = [item(1, categories="Foo|World War II graves in Ukraine", year=1900),
                 item(2, categories="Bar", year=1955),
                 item(3, categories="Bar", year=1954),       # межа не включена: > 1954
                 item(4, categories="Bar", year=None)]       # рік невідомий — не відсіюємо
        res = {e["id"]: [r["code"] for r in e["reasons"]] for e in evaluate(items, t)}
        self.assertEqual(res, {1: ["blacklist"], 2: ["year"], 3: [], 4: []})
        self.assertIn("build_year:1955", reasons_text(evaluate(items, t)[1]["reasons"]))
        # вимкнений фільтр нічого не відсіює
        self.assertEqual([e["reasons"] for e in evaluate(items, Thresholds())], [[], [], [], []])

        h = "ff00ff00ff00ff00"
        dups = [item(1, dhash=h, monument_id="m1"), item(2, dhash=h, monument_id="m2"),
                item(3, dhash=h, monument_id="m1")]
        d_mon = find_duplicates(dups, Thresholds(duplicate_scope="monument"))
        self.assertEqual(set(d_mon), {3})               # 2 — інша пам'ятка, не дубль
        self.assertEqual(set(find_duplicates(dups, Thresholds())), {2, 3})

    def test_years_store_and_old_db_migration(self):
        import sqlite3
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "old.sqlite3"
        con = sqlite3.connect(path)       # база до режимів: runs без campaign/options
        con.executescript("""CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            nomination TEXT NOT NULL, source_type TEXT NOT NULL, source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new', message TEXT, thresholds TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO runs(name,nomination,source_type,source,thresholds,created_at,updated_at)
            VALUES ('старий','quality','folder','/x','{}','t','t');""")
        con.commit(); con.close()
        st = Store(path)
        run = st.list_runs()[0]
        self.assertEqual((run["campaign"], run["options"]), ("wle", {}))
        out = st.load_years({"63-1": 1890, "63-2": "1960", "63-3": "abc", "63-4": 5, "": 1900})
        self.assertEqual((out["loaded"], out["skipped"], out["total"]), (2, 3, 2))
        st.load_years({"63-9": 1800})                      # додає, а не замінює
        self.assertEqual(st.years()["63-1"], 1890)
        self.assertEqual(len(st.years()), 3)
        st.close(); tmp.cleanup()

    def test_api_wlm_flow(self):
        from fastapi.testclient import TestClient
        from wle_filter.server import create_app
        tmp = tempfile.TemporaryDirectory()
        photos = Path(tmp.name) / "photos"
        make_folder(photos)
        store = Store(Path(tmp.name) / "db.sqlite3")
        worker = Worker(store, yolo_loader=lambda: fake_yolo([]), ocr_loader=None)
        client = TestClient(create_app(store, worker))
        files = {"1.jpg": "A/1.jpg", "1 copy.jpg": "A/1_copy.jpg", "1 other.jpg": "B/1_other.jpg",
                 "2.jpg": "B/2.jpg"}

        def fake_list(csv_path, oblast="", limit=None, session=None, progress=None):
            rows = sources.parse_wlm_csv(Path(csv_path).read_text(encoding="utf-8"), oblast)
            return [{"key": r["title"], "title": r["title"], "page_id": None, "author": "x",
                     "timestamp": None, "width": 256, "height": 256, "mime": None, "thumb_url": None,
                     "page_url": None, "local_path": str(photos / files[r["title"][5:]]),
                     "monument_id": r["monument_id"], "categories": r["categories"]} for r in rows]

        body = {"campaign": "wlm", "nomination": "wlm", "source_type": "csv", "source": "x.csv",
                "csv_text": self.CSV, "oblast": "63"}
        # помилки валідації
        self.assertEqual(client.post("/api/runs", json={**body, "nomination": "quality"}).status_code, 400)
        self.assertEqual(client.post("/api/runs", json={**body, "csv_text": "a,b\n1,2"}).status_code, 400)
        self.assertEqual(client.post("/api/runs", json={**body, "oblast": "99"}).status_code, 400)
        self.assertEqual(client.post("/api/runs", json={"campaign": "wle", "nomination": "quality",
                                                         "source_type": "csv", "csv_text": self.CSV}).status_code, 400)
        self.assertEqual(client.post("/api/runs", json={**body, "source_type": "commons",
                                                         "source": "Cat"}).status_code, 400)
        with patch("wle_filter.worker.sources.list_wlm_csv", fake_list):
            r = client.post("/api/runs", json=body)
            self.assertEqual(r.status_code, 201, r.text)
            rid = r.json()["id"]
            worker.join(rid, 30)
        run = client.get(f"/api/runs/{rid}").json()
        self.assertEqual((run["campaign"], run["total"], run["options"]["oblast"]), ("wlm", 4, "63"))
        self.assertTrue(Path(run["options"]["csv_path"]).exists())

        def res(**kw):
            return client.post(f"/api/runs/{rid}/results", json={"view": "all", **kw}).json()

        base = res()
        codes = {i["title"]: [r["code"] for r in i["reasons"]] for i in base["items"]}
        # 1_copy: категорія «Mass grave» + дубль 1.jpg (та сама пам'ятка); 1_other — інша пам'ятка, не дубль
        self.assertEqual(codes["File:1 copy.jpg"], ["blacklist", "duplicate"])
        self.assertEqual(codes["File:1 other.jpg"], [])
        self.assertEqual(client.get("/api/info").json()["years_loaded"], 0)

        # роки підвантажуються «на льоту» і одразу змінюють вердикти
        up = client.post("/api/years", json={"data": {"63-101-0001": 1890, "63-101-0004": 1971}})
        self.assertEqual(up.json()["loaded"], 2)
        after = res()
        codes = {i["title"]: [r["code"] for r in i["reasons"]] for i in after["items"]}
        self.assertEqual(codes["File:2.jpg"], ["year"])
        self.assertEqual(codes["File:1.jpg"], [])
        self.assertEqual(next(i for i in after["items"] if i["title"] == "File:2.jpg")["year"], 1971)
        t = {**after["thresholds"], "year_cutoff": 1990}
        self.assertEqual(next(i for i in res(thresholds=t)["items"] if i["title"] == "File:2.jpg")["reasons"], [])
        self.assertEqual(after["summary"]["reasons"]["year"], 1)
        self.assertEqual(client.get("/api/years").json()["total"], 2)
        self.assertEqual(client.post("/api/years", json={"data": {}}).status_code, 400)

        # експорт містить monument_id і рік
        text = client.get(f"/api/runs/{rid}/export?format=csv").text
        self.assertIn("monument_id", text)
        self.assertIn("build_year:1971", text)
        self.assertEqual(client.delete("/api/years").status_code, 204)
        self.assertEqual(client.get("/api/years").json()["total"], 0)
        store.close(); tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
