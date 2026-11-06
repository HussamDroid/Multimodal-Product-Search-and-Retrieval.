"""Prepare an image-backed OpenCart demo subset without editing the master."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import tempfile

from .catalog import resolve_image
from .importer import normalized, prepare_catalog, read_rows


def prepare_opencart(source, output, image_root, *, duplicates, sheet=None):
    if duplicates not in {'first', 'last', 'error'}:
        raise ValueError('Choose first, last or error for duplicate barcodes')
    rows = read_rows(source, sheet)
    if not rows:
        raise ValueError('The master is empty')
    headers = rows[0]

    def column(name):
        matches = [i for i, h in enumerate(headers) if normalized(h) == normalized(name)]
        if not matches:
            raise ValueError(f'Missing OpenCart column: {name}')
        # OpenCart fields precede warehouse helper columns in this export format.
        return matches[0]

    fields = {key: column(name) for key, name in {
        'barcode': 'Barcode', 'group_id': 'Model Number', 'name_en': 'Product Name',
        'description_en': 'Description', 'price': 'Price', 'stock': 'Quantity',
    }.items()}
    categories = next((i for i, h in enumerate(headers)
                       if normalized(h).startswith('categoriesname')), None)
    enabled = next((i for i, h in enumerate(headers)
                    if normalized(h).startswith('statusenabled')), None)
    language = next((i for i, h in enumerate(headers) if normalized(h) == 'languagecode'), None)
    groups = defaultdict(list)
    for row_number, row in enumerate(rows[1:], 2):
        if not any(row):
            continue
        barcode = row[fields['barcode']].strip()
        if not barcode:
            continue
        cover = resolve_image(image_root, barcode + '/1.jpg')
        if not cover.parent.is_dir() or not any(
            p.is_file() and p.name.casefold() in {'1.jpg', '1.jpeg', '1.png', '1.webp'}
            for p in cover.parent.iterdir()
        ):
            continue
        if language is not None and row[language] and not row[language].lower().startswith('en'):
            raise ValueError(f'Row {row_number}: this adapter requires an English export')
        item = {key: row[i] for key, i in fields.items()}
        item['category'] = row[categories] if categories is not None else ''
        item['enabled'] = row[enabled] if enabled is not None else '1'
        groups[barcode].append((row_number, item))
    selected, audit, disabled = [], [], []
    for barcode, candidates in sorted(groups.items()):
        if len(candidates) > 1 and duplicates == 'error':
            raise ValueError(f'Duplicate barcode {barcode}; select an explicit first/last policy')
        chosen_row, chosen = candidates[-1 if duplicates == 'last' else 0]
        conflicts = {key: list(dict.fromkeys(item[key] for _, item in candidates))
                     for key in candidates[0][1]
                     if len({item[key] for _, item in candidates}) > 1}
        audit.append({'barcode': barcode, 'source_rows': [n for n, _ in candidates],
                      'selected_row': chosen_row, 'conflicts': conflicts})
        if chosen['enabled'] == '0':
            disabled.append(barcode)
            continue
        chosen = dict(chosen)
        chosen.pop('enabled')
        chosen['product_id'] = barcode
        chosen['group_id'] = chosen['group_id'] or barcode
        selected.append(chosen)
    if not selected:
        raise ValueError('No enabled products with matching cover images')
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / 'selected.csv'
        with intermediate.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        report = prepare_catalog(intermediate, output, image_root)
    report.update(source_rows=sum(bool(any(r)) for r in rows[1:]),
                  duplicate_policy=duplicates, selection='enabled products with supplied cover images',
                  source_columns={key: {'heading': headers[i], 'column_number': i+1}
                                  for key, i in fields.items()},
                  barcode_audit=audit, disabled_barcodes=disabled)
    report['notes'] += ['The first OpenCart Barcode column is used, not later warehouse helper columns.',
                        'Barcodes identify products; model numbers provisionally group variants.',
                        'Family assignments and conflicting categories need review before evaluation.']
    output.with_suffix('.import.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
