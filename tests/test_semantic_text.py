from pathlib import Path
import tempfile
import unittest

import numpy as np

from souqlens.encoders import ClipEncoder
from souqlens.index import VectorIndex
from test_retrieval import fixture


class SemanticTextTests(unittest.TestCase):
    def engine(self):
        engine = fixture()
        engine.index.metadata = {**engine.index.metadata, 'schema_version': 2,
                                 'encoder': {'backend': 'test-only', 'semantic_model': 'fake'}}
        engine.index = VectorIndex(engine.products, engine.index.text_vectors,
            engine.index.image_vectors, engine.index.has_image, engine.index.metadata,
            semantic_vectors=np.array([[0, 1, 0], [1, 0, 0]]))
        engine.encoder.identity = dict(engine.index.metadata['encoder'])
        engine.encoder.semantic_queries = lambda texts: np.array([[0, 1, 0] for _ in texts])
        return engine

    def test_text_uses_separate_space_and_photo_search_is_preserved(self):
        engine = self.engine()
        # The legacy text encoder would rank the lamp for this unseen Arabic query.
        self.assertEqual(engine.search('وعاء للمطبخ', mode='dense')['results'][0]['product']['product_id'], 'jar')
        self.assertEqual(engine.search(image=Path('photo'), mode='dense')['results'][0]['product']['product_id'], 'lamp')
        self.assertEqual(engine.search('وعاء', image=Path('photo'), image_weight=0,
                                       mode='dense')['results'][0]['product']['product_id'], 'jar')
        self.assertEqual(engine.search('وعاء', image=Path('photo'), image_weight=1,
                                       mode='dense')['results'][0]['product']['product_id'], 'lamp')

    def test_semantic_roundtrip_and_missing_arrays_are_rejected(self):
        index = self.engine().index
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'index'
            index.save(path)
            loaded = VectorIndex.load(path)
            np.testing.assert_allclose(loaded.semantic_vectors, index.semantic_vectors)
        with self.assertRaisesRegex(ValueError, 'missing semantic'):
            VectorIndex(index.products, index.text_vectors, index.image_vectors,
                        index.has_image, index.metadata)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            VectorIndex(index.products, index.text_vectors, index.image_vectors,
                        index.has_image, index.metadata, np.zeros((2, 3)))

    def test_bad_semantic_query_shape_or_zero_norm_is_rejected(self):
        for vector in [np.array([[1, 0]]), np.zeros((1, 3))]:
            engine = self.engine()
            engine.encoder.semantic_queries = lambda texts: vector
            with self.assertRaisesRegex(ValueError, 'Semantic query'):
                engine.search('jar', mode='dense')

    def test_e5_uses_distinct_query_and_document_prefixes(self):
        class Model:
            def encode(self, texts, **kwargs):
                self.texts = texts
                return np.ones((len(texts), 3))
        encoder = ClipEncoder.__new__(ClipEncoder)
        encoder.semantic_model, encoder.batch_size = Model(), 4
        encoder.semantic_queries(['مروحة'])
        self.assertEqual(encoder.semantic_model.texts, ['query: مروحة'])
        encoder.semantic_documents(['Desk fan'])
        self.assertEqual(encoder.semantic_model.texts, ['passage: Desk fan'])

    def test_build_keeps_semantic_and_visual_dimensions_separate(self):
        engine = self.engine()
        engine.encoder.semantic_documents = lambda texts: np.array([[0, 1, 0], [1, 0, 0]])
        index = VectorIndex.build(engine.products, Path('.'), engine.encoder)
        self.assertEqual(index.metadata['schema_version'], 2)
        self.assertEqual(index.text_vectors.shape, (2, 2))
        self.assertEqual(index.semantic_vectors.shape, (2, 3))

    def test_unknown_semantic_model_is_rejected_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported semantic'):
            ClipEncoder(semantic_text=True, semantic_model='unknown/model')

    def test_cli_restores_the_model_and_revisions_from_the_index(self):
        from unittest.mock import patch
        from souqlens.cli import indexed_engine
        engine = self.engine()
        engine.index.metadata['encoder'].update(text_revision='0'*40, image_revision='1'*40,
                                                semantic_revision='2'*40)
        engine.encoder.identity = dict(engine.index.metadata['encoder'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'index'
            engine.index.save(path)
            with patch('souqlens.encoders.ClipEncoder', return_value=engine.encoder) as constructor:
                loaded = indexed_engine(path)
                loaded._encoder()
                self.assertEqual(constructor.call_args.kwargs['semantic_model'], 'fake')
                self.assertEqual(constructor.call_args.kwargs['semantic_revision'], '2'*40)
