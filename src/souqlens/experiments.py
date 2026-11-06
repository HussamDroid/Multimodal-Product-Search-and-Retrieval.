"""Compare versioned indexes on the same judged text-query splits."""

import gc
import os
import platform
from pathlib import Path

from .catalog import fingerprint
from .evaluation import evaluate, load_queries


def compare_indexes(indexes, queries_path, image_root, *, splits=('dev', 'test'), k=3, device='cpu'):
    from .cli import indexed_engine
    reports, catalog_id = {}, None
    for path in indexes:
        path = Path(path)
        if path.name in reports:
            raise ValueError('Index directory names must be unique in a comparison')
        engine = indexed_engine(path, device)
        current = fingerprint(engine.products)
        if catalog_id is not None and current != catalog_id:
            raise ValueError('Compare indexes built from the same catalog')
        catalog_id = current
        queries = load_queries(queries_path, engine.products)
        if any(q['modality'] != 'text' for q in queries if q['split'] in splits):
            raise ValueError('This comparison requires a shared text-only query set')
        reports[path.name] = {
            split: {mode: evaluate(engine, queries, image_root=image_root, mode=mode,
                                   split=split, k=k) for mode in ('bm25', 'dense', 'hybrid')}
            for split in splits
        }
        del engine
        gc.collect()
    return {'catalog_sha256': catalog_id, 'device': device, 'machine': platform.machine(),
            'python': platform.python_version(), 'omp_threads': os.getenv('OMP_NUM_THREADS'),
            'indexes': reports,
            'notes': ['Compare modes on the same text queries; image queries require a separate report.',
                      'Owner review is required before calling draft labels a benchmark.',
                      'Timings use the recorded device, exclude one warm-up per mode and do not measure hosted queue time.']}
