from pathlib import Path
import tempfile
import unittest

from souqlens.catalog import load_catalog
from souqlens.importer import plain_text, prepare_catalog


class ImportTests(unittest.TestCase):
    def test_maps_master_fields_and_matches_barcode_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"00123").mkdir()
            (root/"00123/1.JPG").write_bytes(b"fixture")
            source = root/"master.csv"
            source.write_text('Model Number,Barcode,Product Name,Description,Price,Quantity\nM-1,00123,Jar,<p>Glass jar</p>,25,10\n')
            output = root/"prepared.csv"
            report = prepare_catalog(source, output, root)
            product = load_catalog(output)[0]
            self.assertEqual(product.barcode, "00123")
            self.assertEqual(product.description_en, "Glass jar")
            self.assertEqual(product.image, "00123/1.JPG")
            self.assertEqual(report["matched_cover_images"], 1)
            self.assertEqual(report["groups_defaulted_to_product_id"], 1)
            self.assertTrue(source.exists())

    def test_bad_numeric_data_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"master.csv"
            source.write_text('ID,Name,Price,Quantity\np1,Jar,invalid,5\n')
            with self.assertRaises(ValueError):
                prepare_catalog(source, root/"output.csv", root)
            self.assertFalse((root/"output.csv").exists())

    def test_scientific_barcode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"master.csv"
            source.write_text('Barcode,Name,Price,Quantity\n1.234E+12,Jar,25,5\n')
            with self.assertRaisesRegex(ValueError, "scientific notation"):
                prepare_catalog(source,root/"output.csv",root)

    def test_html_content_is_cleaned(self):
        self.assertEqual(plain_text('<p>Glass&nbsp;jar</p><script>bad()</script><li>1000 ml</li>'), 'Glass jar 1000 ml')

    def test_excel_barcode_number_format_is_preserved(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest('Excel extra is not installed')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            book=Workbook();sheet=book.active
            sheet.append(['Barcode','Name','Price','Quantity'])
            sheet.append([123,'Jar',25,5]);sheet['A2'].number_format='00000'
            book.save(root/'master.xlsx');book.close()
            prepare_catalog(root/'master.xlsx',root/'output.csv',root)
            self.assertEqual(load_catalog(root/'output.csv')[0].barcode,'00123')
