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
        self.assertGreaterEqual(Thresholds.from_dict({"person_confidence": 0}).person_confidence,
                                config.PERSON_CONF_FLOOR)

    def test_presets_have_both_nominations(self):
        p = config.load_presets()
        self.assertIn("quality", p)
        self.assertIn("quantity", p)
        self.assertEqual(p["quantity"]["thresholds"]["duplicate_scope"], "author")

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

    def test_person_detections_keep_low_conf(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = analysis.person_detections(img, fake_yolo([(0, 0.1, [0, 0, 50, 50]), (0, 0.01, [0, 0, 90, 90])]))
        self.assertEqual([d["conf"] for d in dets], [0.1])     # 0.01 < floor

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
        self.assertEqual(set(m), {"blur", "persons", "sign", "dhash", "phash"})
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
        w = Worker(self.store, yolo_loader=lambda: fake_yolo([]))
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

    def test_quality_vs_quantity_duplicates(self):
        run_id = self.run_folder()
        imgs = self.store.images(run_id)
        names = {i["id"]: i["key"] for i in imgs}
        q = evaluate(imgs, Thresholds.from_dict(config.load_presets()["quality"]["thresholds"]))
        qn = evaluate(imgs, Thresholds.from_dict(config.load_presets()["quantity"]["thresholds"]))
        dup = lambda ev: {names[e["id"]] for e in ev if any(r["code"] == "duplicate" for r in e["reasons"])}
        self.assertEqual(len(dup(q)), 2)                 # серед усіх: 2 копії
        self.assertEqual(len(dup(qn)), 1)                # лише в межах автора

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
        w = Worker(self.store, yolo_loader=lambda: slow)
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
        cls.worker = Worker(cls.store, yolo_loader=lambda: fake_yolo([]))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
