import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from souqlens.experiments import compare_indexes
from souqlens.evaluation import metrics
from test_retrieval import fixture


class ExperimentTests(unittest.TestCase):
    def queries(self, root):
        path = root/'queries.jsonl'
        rows = [{'query_id': 'dev', 'text': 'jar', 'language': 'en', 'modality': 'text',
                 'split': 'dev', 'judgment_status': 'draft', 'relevance': {'jar': 3, 'lamp': 0}},
                {'query_id': 'test', 'text': 'مصباح', 'language': 'ar', 'modality': 'text',
                 'split': 'test', 'judgment_status': 'reviewed', 'relevance': {'jar': 0, 'lamp': 3}}]
        path.write_text('\n'.join(json.dumps(q) for q in rows))
        return path

    def test_modes_share_queries_and_report_review_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('souqlens.cli.indexed_engine', side_effect=lambda *a: fixture()):
                report = compare_indexes([Path('v1'), Path('v2')], self.queries(root), root, k=1)
            for experiment in report['indexes'].values():
                for mode in ['bm25', 'dense', 'hybrid']:
                    self.assertEqual(experiment['dev'][mode]['per_query'][0]['query_id'], 'dev')
                    self.assertEqual(experiment['test'][mode]['judgment_review']['reviewed'], 1)
                    self.assertEqual(experiment['dev'][mode]['judgment_review']['draft'], 1)
            self.assertEqual(metrics(['wrong', 'jar'], {'jar': 3}, 2)['hit@1'], 0)
            self.assertEqual(metrics(['jar'], {'jar': 3}, 2)['hit@1'], 1)

    def test_changed_catalog_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = fixture(); other.products = list(reversed(other.products))
            with patch('souqlens.cli.indexed_engine', side_effect=[fixture(), other]):
                with self.assertRaisesRegex(ValueError, 'same catalog'):
                    compare_indexes([Path('v1'), Path('v2')], self.queries(root), root, k=1)
