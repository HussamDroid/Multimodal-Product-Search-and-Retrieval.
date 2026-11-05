import tempfile
import unittest
from pathlib import Path

from souqlens.catalog import load_catalog, resolve_image
from souqlens.lexical import BM25, tokenize


class BaselineTests(unittest.TestCase):
    def test_english_and_arabic_search(self):
        index = BM25(["glass jar برطمان زجاجي", "table lamp مصباح طاولة"])
        self.assertEqual(index.scores("زجاجي").argmax(), 0)
        self.assertEqual(index.scores("table lamp").argmax(), 1)
        self.assertTrue((index.scores("unseenword") == 0).all())

    def test_arabic_diacritics_and_alef(self):
        self.assertEqual(tokenize("إِضَاءَة"), tokenize("اضاءة"))

    def test_csv_keeps_barcode_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.csv"
            path.write_text("product_id,group_id,barcode,name_en,price,stock\np1,g1,00123,Jar,20,0\n")
            self.assertEqual(load_catalog(path)[0].barcode, "00123")

    def test_bad_identifiers_and_numbers_fail(self):
        for rows in ["p1,g1,Jar,nan,0", "p1,g1,Jar,20,-1", "p1,g1,Jar,20,1\np1,g2,Other,10,1"]:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "catalog.csv"
                path.write_text("product_id,group_id,name_en,price,stock\n" + rows + "\n")
                with self.assertRaises(ValueError):
                    load_catalog(path)

    def test_image_root_cannot_be_escaped(self):
        with self.assertRaises(ValueError):
            resolve_image(Path("data/sample"), "../../../secret.jpg")


if __name__ == "__main__":
    unittest.main()
