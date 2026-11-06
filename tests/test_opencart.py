import csv
from pathlib import Path
import tempfile
import unittest

from souqlens.catalog import load_catalog
from souqlens.opencart import prepare_opencart


class OpenCartTests(unittest.TestCase):
    def master(self, root):
        for barcode in ['00123', '00456']:
            (root/barcode).mkdir()
            (root/barcode/'1.jpg').write_bytes(b'fixture')
        source = root/'master.csv'
        with source.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Product ID', 'Product Name', 'Model Number', 'Language Code',
                             'Description', 'Barcode', 'Price', 'Quantity',
                             'Status (Enabled=1/Disabled=0)', 'Categories Name Category 1', 'Barcode'])
            writer.writerows([
                ['', 'Blue rug', 'Shared model', 'en-gb', '<p>Blue</p>', '00123', '10', '4', '1', 'Rugs', 'wrong'],
                ['', 'Blue carpet', 'Shared model', 'en-gb', '<p>Blue carpet</p>', '00123', '10', '3', '1', '', ''],
                ['', 'Red rug', 'Shared model', 'en-gb', 'Red', '00456', '12', '0', '1', 'Rugs', ''],
                ['', 'No cover', 'M3', 'en-gb', 'Unmatched', '00789', '2', '1', '1', '', ''],
            ])
        return source

    def test_last_row_policy_retains_variants_and_reports_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = self.master(root)
            report = prepare_opencart(source, root/'catalog.csv', root, duplicates='last')
            products = load_catalog(root/'catalog.csv', root)
            self.assertEqual([p.product_id for p in products], ['00123', '00456'])
            self.assertEqual(products[0].stock, 3)
            self.assertEqual(products[0].name_en, 'Blue carpet')
            self.assertEqual(products[0].category, '')
            self.assertEqual(products[0].group_id, products[1].group_id)
            self.assertEqual(report['barcode_audit'][0]['selected_row'], 3)
            self.assertEqual(report['barcode_audit'][0]['conflicts']['stock'], ['4', '3'])
            self.assertEqual(report['matched_cover_images'], 2)

    def test_duplicate_error_leaves_no_output_and_first_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = self.master(root)
            output = root/'catalog.csv'
            with self.assertRaisesRegex(ValueError, 'Duplicate barcode'):
                prepare_opencart(source, output, root, duplicates='error')
            self.assertFalse(output.exists())
            prepare_opencart(source, output, root, duplicates='first')
            self.assertEqual(load_catalog(output)[0].stock, 4)
