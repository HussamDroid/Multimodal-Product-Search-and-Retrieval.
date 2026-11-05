import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from souqlens.catalog import Product, fingerprint
from souqlens.hosting import export_bundle, validate_bundle, stage_space
from souqlens.index import VectorIndex


class BundleTests(unittest.TestCase):
    def setup_index(self, root):
        (root/'photos/00123').mkdir(parents=True)
        (root/'photos/00123/1.jpg').write_bytes(b'fixture image bytes')
        product=Product('p1','g1','00123','Jar','','','','Kitchen',25,5,'00123/1.jpg')
        digest=hashlib.sha256((root/'photos/00123/1.jpg').read_bytes()).hexdigest()
        index=VectorIndex([product],np.array([[1,0]]),np.array([[1,0]]),[True],
            {'schema_version':1,'catalog_sha256':fingerprint([product]),'encoder':{'backend':'test-only'},
             'image_sha256':{'p1':digest}})
        index.save(root/'index')

    def test_bundle_is_portable_and_checksums_detect_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.setup_index(root)
            report=export_bundle(root/'index',root/'photos',root/'bundle')
            self.assertEqual(report['products'],1)
            self.assertEqual(validate_bundle(root/'bundle')['products'],1)
            (root/'bundle/images/00123/1.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                validate_bundle(root/'bundle')

    def test_changed_source_image_requires_reindexing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.setup_index(root)
            (root/'photos/00123/1.jpg').write_bytes(b'updated image')
            with self.assertRaisesRegex(ValueError,'changed since indexing'):
                export_bundle(root/'index',root/'photos',root/'bundle')
            self.assertFalse((root/'bundle').exists())

    def test_bundle_manifest_cannot_reference_outside_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.setup_index(root)
            export_bundle(root/'index',root/'photos',root/'bundle')
            path=root/'bundle/bundle.json'
            data=json.loads(path.read_text());data['files']['../../outside']='x'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                validate_bundle(root/'bundle')

    def test_existing_bundle_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.setup_index(root)
            export_bundle(root/'index',root/'photos',root/'bundle')
            with self.assertRaisesRegex(ValueError,'already exists'):
                export_bundle(root/'index',root/'photos',root/'bundle')

    def test_space_staging_does_not_copy_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['deploy/Dockerfile','deploy/README.md','pyproject.toml','src/souqlens/api.py',
                         'data/private/master.csv','artifacts/vectors.npz','.env']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            stage_space(root,root/'staged')
            self.assertTrue((root/'staged/Dockerfile').is_file())
            self.assertTrue((root/'staged/src/souqlens/api.py').is_file())
            self.assertFalse((root/'staged/data').exists())
            self.assertFalse((root/'staged/.env').exists())
