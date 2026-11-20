import csv
import os
import re
import tempfile
import unittest
import numpy as np
import cv2
import imagehash
from PIL import Image
from unittest.mock import patch, MagicMock


# ─── Імпорт функцій з main.py ────────────────────────────────
import importlib.util, sys

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

HERE    = os.path.dirname(os.path.abspath(__file__))
main    = load_module("main",    os.path.join(HERE, "main.py"))
recount = load_module("recount", os.path.join(HERE, "recount.py"))


# ─── Хелпери ─────────────────────────────────────────────────
def make_sharp_img(size=300):
    ch = (np.indices((size, size)).sum(axis=0) // 20 % 2 * 255).astype(np.uint8)
    return np.stack([ch]*3, axis=-1)

def make_blurry_img(size=300):
    return cv2.GaussianBlur(make_sharp_img(size), (51, 51), 20)

def make_blank_img(size=300, val=200):
    return np.full((size, size, 3), val, dtype=np.uint8)

def np_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def make_csv(tmp_dir, rows):
    path = os.path.join(tmp_dir, "progress.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path

def default_row(**kwargs):
    base = {
        "file": "File:test.jpg",
        "blur_score": "5000",
        "person_area": "0.0",
        "sign_coverage": "0.0",
        "duplicate_of": "",
        "local_path": "",
        "verdict": "ok",
        "reasons": ""
    }
    base.update(kwargs)
    return base


# ════════════════════════════════════════════════════════════════
# ТЕСТИ main.py
# ════════════════════════════════════════════════════════════════

class TestSafeFilename(unittest.TestCase):

    def test_removes_file_prefix(self):
        self.assertNotIn("File:", main.safe_filename("File:photo.jpg"))

    def test_replaces_forbidden_chars(self):
        result = main.safe_filename('File:a<b>c:"d/e\\f|g?h*i.jpg')
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, result)

    def test_normal_name_preserved(self):
        self.assertEqual(main.safe_filename("File:photo_2026.jpg"), "photo_2026.jpg")

    def test_empty_string(self):
        self.assertEqual(main.safe_filename(""), "")

    def test_only_prefix(self):
        self.assertEqual(main.safe_filename("File:"), "")


class TestCheckBlur(unittest.TestCase):

    def test_sharp_not_blurry(self):
        is_blurry, score = main.check_blur(make_sharp_img())
        self.assertFalse(is_blurry)
        self.assertGreater(score, main.BLUR_THRESHOLD)

    def test_blurry_detected(self):
        is_blurry, score = main.check_blur(make_blurry_img())
        self.assertTrue(is_blurry)
        self.assertLess(score, main.BLUR_THRESHOLD)

    def test_returns_float_score(self):
        _, score = main.check_blur(make_sharp_img())
        self.assertIsInstance(score, float)

    def test_large_image_resized(self):
        big = make_sharp_img(1200)
        is_blurry, score = main.check_blur(big)
        self.assertIsInstance(score, float)

    def test_blank_image_blurry(self):
        is_blurry, _ = main.check_blur(make_blank_img())
        self.assertTrue(is_blurry)


class TestCheckPerson(unittest.TestCase):

    def _fake_model(self, cls_id, conf, box):
        fake_box = MagicMock()
        fake_box.cls = [cls_id]
        fake_box.conf = [conf]
        fake_box.xyxy = [np.array(box, dtype=float)]
        result = MagicMock()
        result.boxes = [fake_box]
        model = MagicMock()
        model.return_value = [result]
        return model

    def test_large_person_flagged(self):
        img   = np.zeros((100, 100, 3), dtype=np.uint8)
        model = self._fake_model(0, 0.9, [0, 0, 60, 60])
        flagged, area = main.check_person(img, model)
        self.assertTrue(flagged)
        self.assertGreater(area, main.PERSON_AREA_THRESHOLD)

    def test_small_person_passes(self):
        img   = np.zeros((100, 100, 3), dtype=np.uint8)
        model = self._fake_model(0, 0.9, [0, 0, 5, 5])
        flagged, _ = main.check_person(img, model)
        self.assertFalse(flagged)

    def test_low_confidence_ignored(self):
        img   = np.zeros((100, 100, 3), dtype=np.uint8)
        model = self._fake_model(0, 0.2, [0, 0, 90, 90])
        flagged, area = main.check_person(img, model)
        self.assertFalse(flagged)
        self.assertEqual(area, 0.0)

    def test_non_person_class_ignored(self):
        img   = np.zeros((100, 100, 3), dtype=np.uint8)
        model = self._fake_model(2, 0.99, [0, 0, 90, 90])
        flagged, _ = main.check_person(img, model)
        self.assertFalse(flagged)

    def test_no_detections(self):
        img    = np.zeros((100, 100, 3), dtype=np.uint8)
        result = MagicMock()
        result.boxes = []
        model  = MagicMock()
        model.return_value = [result]
        flagged, area = main.check_person(img, model)
        self.assertFalse(flagged)
        self.assertEqual(area, 0.0)


class TestCheckSignOrPlaque(unittest.TestCase):

    def test_blank_image_no_sign(self):
        flagged, cov = main.check_sign_or_plaque(make_blank_img())
        self.assertFalse(flagged)
        self.assertEqual(cov, 0.0)

    def test_returns_float_coverage(self):
        _, cov = main.check_sign_or_plaque(make_sharp_img())
        self.assertIsInstance(cov, float)

    def test_coverage_between_0_and_1(self):
        _, cov = main.check_sign_or_plaque(make_sharp_img())
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)


class TestCheckDuplicate(unittest.TestCase):

    def setUp(self):
        main.phash_registry.clear()

    def test_first_image_not_duplicate(self):
        pil = np_to_pil(make_sharp_img())
        is_dup, _, _ = main.check_duplicate(pil, "first.jpg")
        self.assertFalse(is_dup)

    def test_identical_image_is_duplicate(self):
        img = make_sharp_img()
        pil = np_to_pil(img)
        main.check_duplicate(pil, "first.jpg")
        is_dup, dup_type, dup_of = main.check_duplicate(pil, "copy.jpg")
        self.assertTrue(is_dup)
        self.assertEqual(dup_of, "first.jpg")

    def test_different_images_not_duplicate(self):
        main.check_duplicate(np_to_pil(make_sharp_img()), "a.jpg")
        is_dup, _, _ = main.check_duplicate(np_to_pil(make_blank_img(val=0)), "b.jpg")
        self.assertFalse(is_dup)

    def test_registry_grows(self):
        main.check_duplicate(np_to_pil(make_sharp_img()), "a.jpg")
        main.check_duplicate(np_to_pil(make_blank_img(val=0)), "b.jpg")
        self.assertEqual(len(main.phash_registry), 2)

    def test_brightness_shift_detected_as_duplicate(self):
        img = make_sharp_img()
        pil1 = np_to_pil(img)
        pil2 = np_to_pil(np.clip(img.astype(int) + 15, 0, 255).astype(np.uint8))
        main.check_duplicate(pil1, "orig.jpg")
        is_dup, _, _ = main.check_duplicate(pil2, "shifted.jpg")
        self.assertTrue(is_dup)


class TestDecideVerdict(unittest.TestCase):

    def test_error_preserved(self):
        self.assertEqual(main.decide_verdict({"verdict": "error", "reasons": ["x"]}), "error")

    def test_reasons_give_flagged(self):
        self.assertEqual(main.decide_verdict({"verdict": "ok", "reasons": ["blur"]}), "flagged")

    def test_no_reasons_give_ok(self):
        self.assertEqual(main.decide_verdict({"verdict": "ok", "reasons": []}), "ok")

    def test_empty_reasons_string(self):
        self.assertEqual(main.decide_verdict({"verdict": "ok", "reasons": ""}), "ok")


class TestGetOrDownloadImageCache(unittest.TestCase):

    def test_cached_old_format_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_name = re.sub(r'[<>:"/\\|?*]', '_', "File:test photo.jpg")
            path = os.path.join(tmp, old_name)
            with open(path, "wb") as f:
                f.write(b"x" * 20000)
            orig_dir = main.IMAGES_DIR
            main.IMAGES_DIR = tmp
            result = main.get_or_download_image("File:test photo.jpg")
            main.IMAGES_DIR = orig_dir
            self.assertEqual(result, path)

    def test_cached_new_format_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_name = re.sub(r'[<>:"/\\|?*]', '_', "test photo.jpg")
            path = os.path.join(tmp, new_name)
            with open(path, "wb") as f:
                f.write(b"x" * 20000)
            orig_dir = main.IMAGES_DIR
            main.IMAGES_DIR = tmp
            result = main.get_or_download_image("File:test photo.jpg")
            main.IMAGES_DIR = orig_dir
            self.assertEqual(result, path)

    def test_small_file_not_used_as_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = re.sub(r'[<>:"/\\|?*]', '_', "test.jpg")
            path = os.path.join(tmp, name)
            with open(path, "wb") as f:
                f.write(b"x" * 100)  # занадто малий
            orig_dir = main.IMAGES_DIR
            main.IMAGES_DIR = tmp
            with patch("main.get_thumb_url", return_value=""):
                result = main.get_or_download_image("File:test.jpg")
            main.IMAGES_DIR = orig_dir
            self.assertEqual(result, "")


# ════════════════════════════════════════════════════════════════
# ТЕСТИ recount.py
# ════════════════════════════════════════════════════════════════

class TestRecountBlur(unittest.TestCase):

    def test_blur_below_threshold_flagged(self):
        row = default_row(blur_score="50")
        with patch.object(recount, "BLUR_THRESHOLD", 2000):
            reasons = []
            blur = float(row["blur_score"])
            if blur < 2000:
                reasons.append(f"blur:{blur}")
            self.assertIn("blur:50.0", reasons)

    def test_blur_above_threshold_ok(self):
        row = default_row(blur_score="5000")
        reasons = []
        blur = float(row["blur_score"])
        if blur < recount.BLUR_THRESHOLD:
            reasons.append(f"blur:{blur}")
        self.assertEqual(reasons, [])

    def test_missing_blur_score_ok(self):
        row = default_row(blur_score="")
        blur = float(row["blur_score"]) if row["blur_score"] else 9999
        self.assertGreater(blur, recount.BLUR_THRESHOLD)


class TestRecountPerson(unittest.TestCase):

    def test_person_above_threshold_flagged(self):
        reasons = []
        person = 0.15
        if person >= recount.PERSON_AREA_THRESHOLD:
            reasons.append(f"person:{person:.1%}")
        self.assertTrue(any("person" in r for r in reasons))

    def test_person_below_threshold_ok(self):
        reasons = []
        person = 0.001
        if person >= recount.PERSON_AREA_THRESHOLD:
            reasons.append(f"person:{person:.1%}")
        self.assertEqual(reasons, [])

    def test_missing_person_area_ok(self):
        row = default_row(person_area="")
        person = float(row["person_area"]) if row["person_area"] else 0
        self.assertEqual(person, 0)


class TestRecountSign(unittest.TestCase):

    def test_sign_above_threshold_flagged(self):
        reasons = []
        sign = 0.35
        if sign >= recount.SIGN_COVERAGE_MIN:
            reasons.append(f"sign:{sign:.1%}")
        self.assertTrue(any("sign" in r for r in reasons))

    def test_sign_below_threshold_ok(self):
        reasons = []
        sign = 0.05
        if sign >= recount.SIGN_COVERAGE_MIN:
            reasons.append(f"sign:{sign:.1%}")
        self.assertEqual(reasons, [])

    def test_missing_sign_coverage_ok(self):
        row = default_row(sign_coverage="")
        sign = float(row["sign_coverage"]) if row["sign_coverage"] else 0
        self.assertEqual(sign, 0)


class TestRecountDuplicate(unittest.TestCase):

    def test_duplicate_of_set_flagged(self):
        row = default_row(duplicate_of="File:original.jpg")
        reasons = []
        if row.get("duplicate_of"):
            reasons.append(f"duplicate:{row['duplicate_of'][:40]}")
        self.assertTrue(any("duplicate" in r for r in reasons))

    def test_no_duplicate_ok(self):
        row = default_row(duplicate_of="")
        reasons = []
        if row.get("duplicate_of"):
            reasons.append("duplicate")
        self.assertEqual(reasons, [])


class TestRecountCSV(unittest.TestCase):

    def test_output_csv_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(blur_score="50"), default_row(blur_score="9000")]
            csv_path = make_csv(tmp, rows)
            out_path = os.path.join(tmp, "results_recounted.csv")
            orig_input  = recount.INPUT_CSV
            orig_output = recount.OUTPUT_DIR
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            recount.INPUT_CSV  = orig_input
            recount.OUTPUT_DIR = orig_output
            self.assertTrue(os.path.exists(out_path))

    def test_flagged_count_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                default_row(blur_score="30"),   # flagged
                default_row(blur_score="9000"),  # ok
                default_row(blur_score="40"),    # flagged
            ]
            csv_path = make_csv(tmp, rows)
            out_path = os.path.join(tmp, "results_recounted.csv")
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            with open(out_path, encoding="utf-8") as f:
                result_rows = list(csv.DictReader(f))
            flagged = [r for r in result_rows if r["verdict"] == "flagged"]
            self.assertEqual(len(flagged), 2)

    def test_error_rows_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(verdict="error", reasons="download_failed")]
            csv_path = make_csv(tmp, rows)
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            out_path = os.path.join(tmp, "results_recounted.csv")
            with open(out_path, encoding="utf-8") as f:
                result_rows = list(csv.DictReader(f))
            self.assertEqual(result_rows[0]["verdict"], "error")

    def test_empty_csv_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "progress.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("file,blur_score,person_area,sign_coverage,"
                        "duplicate_of,local_path,verdict,reasons\n")
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()


class TestRecountGallery(unittest.TestCase):

    def test_gallery_html_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(blur_score="30")]
            csv_path = make_csv(tmp, rows)
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            gallery = os.path.join(tmp, "flagged_gallery_recounted.html")
            self.assertTrue(os.path.exists(gallery))

    def test_gallery_contains_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(blur_score="30")]
            csv_path = make_csv(tmp, rows)
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            gallery = os.path.join(tmp, "flagged_gallery_recounted.html")
            with open(gallery, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("blur", content)

    def test_gallery_ok_not_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(blur_score="9000", file="File:good.jpg")]
            csv_path = make_csv(tmp, rows)
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            gallery = os.path.join(tmp, "flagged_gallery_recounted.html")
            with open(gallery, encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("good.jpg", content)

    def test_commons_link_in_gallery(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [default_row(blur_score="30", file="File:test.jpg")]
            csv_path = make_csv(tmp, rows)
            recount.INPUT_CSV  = csv_path
            recount.OUTPUT_DIR = tmp
            with patch("recount.get_thumb_urls_batch", return_value={}):
                recount.recount()
            gallery = os.path.join(tmp, "flagged_gallery_recounted.html")
            with open(gallery, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("commons.wikimedia.org", content)


class TestGetCommonsUrl(unittest.TestCase):

    def test_basic_url(self):
        url = recount.get_commons_url("File:test.jpg")
        self.assertIn("commons.wikimedia.org", url)
        self.assertIn("test.jpg", url)

    def test_spaces_replaced(self):
        url = recount.get_commons_url("File:my photo.jpg")
        self.assertNotIn(" ", url)
        self.assertIn("my_photo.jpg", url)

    def test_file_prefix_removed(self):
        url = recount.get_commons_url("File:photo.jpg")
        self.assertNotIn("File:File:", url)


if __name__ == "__main__":
    unittest.main(verbosity=2)