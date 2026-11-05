"""Convert an existing product export into the search catalog schema."""

import csv
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from .catalog import Product, load_catalog, resolve_image

ALIASES = {
    "product_id": ["product_id", "product id", "id", "model number", "model", "sku"],
    "group_id": ["group_id", "product family", "family id", "variant group"],
    "barcode": ["barcode", "ean", "gtin"],
    "name_en": ["name_en", "english name", "product name english", "product name", "name"],
    "name_ar": ["name_ar", "arabic name", "product name arabic", "arabic product name"],
    "description_en": ["description_en", "english description", "description english", "description"],
    "description_ar": ["description_ar", "arabic description", "description arabic"],
    "category": ["category", "minimall category", "category name", "category/subcategory"],
    "price": ["price", "selling price", "sale price"],
    "stock": ["stock", "quantity", "qty", "stock quantity"],
    "image": ["image", "product image", "image path", "cover image"],
}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "li", "br", "tr", "td", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden-1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain_text(value):
    parser = PlainText()
    parser.feed(unescape(str(value)))
    return " ".join("".join(parser.parts).split())


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def read_rows(source: Path, sheet=None):
    if source.suffix.lower() == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as error:
            raise RuntimeError("Install Excel support with pip install -e '.[import]'") from error
        workbook = load_workbook(source, read_only=True, data_only=True)
        try:
            worksheet = workbook[sheet] if sheet else workbook.active
            def value(cell):
                result = cell.value
                if result is None:
                    return ""
                if isinstance(result, (int, float)) and not isinstance(result, bool):
                    if result == int(result):
                        number_format = cell.number_format or ""
                        if re.fullmatch(r"0+", number_format):
                            return str(int(result)).zfill(len(number_format))
                        return str(int(result))
                return str(result).strip()
            return [[value(cell) for cell in row] for row in worksheet.iter_rows()]
        finally:
            workbook.close()
    if source.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Use an .xlsx, UTF-8 .csv or .tsv product export")
    with source.open(encoding="utf-8-sig", newline="") as file:
        sample = file.read(8192)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        return list(csv.reader(file, dialect))


def prepare_catalog(source: Path, output: Path, image_root: Path, mapping=None, sheet=None):
    if output.exists() or output.with_suffix(".import.json").exists():
        raise ValueError("Output already exists; use a new output filename")
    rows = read_rows(source, sheet)
    if len(rows) < 2:
        raise ValueError("The export needs column headings and product rows")
    headers = [str(h).strip() for h in rows[0]]
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate column headings; rename or remove duplicate columns first")
    by_name = {normalized(h): h for h in headers}
    selected = {}
    for field, aliases in ALIASES.items():
        selected[field] = next((by_name[normalized(a)] for a in aliases if normalized(a) in by_name), None)
    for field, column in (mapping or {}).items():
        if field not in ALIASES or column is not None and column not in headers:
            raise ValueError(f"Unknown field or source column in mapping: {field}")
        selected[field] = column
    missing = [f for f in ["name_en", "price", "stock"] if not selected[f]]
    if missing or not (selected["product_id"] or selected["barcode"]):
        raise ValueError("Map name_en, price, stock and either product_id or barcode. Missing: " + ", ".join(missing))
    converted, missing_images, inferred_groups = [], [], 0
    for row_number, row in enumerate(rows[1:], 2):
        if not any(str(v).strip() for v in row):
            continue
        record = dict(zip(headers, row))
        item = {field: str(record.get(column, "") or "").strip() if column else ""
                for field, column in selected.items()}
        for field in ["barcode", "product_id"]:
            if re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", item[field]):
                raise ValueError(f"Row {row_number}: {field} uses scientific notation; restore its full text value")
        item["product_id"] = item["product_id"] or item["barcode"]
        if not item["group_id"]:
            item["group_id"] = item["product_id"]
            inferred_groups += 1
        for field in ["name_en", "name_ar", "description_en", "description_ar", "category"]:
            item[field] = plain_text(item[field])
        # Existing export paths often refer to a different server; barcode folders are authoritative here.
        image = None
        if item["barcode"]:
            candidate = resolve_image(image_root, item["barcode"]+"/1.jpg")
            if candidate.parent.is_dir():
                priority = {"1.jpg": 0, "1.jpeg": 1, "1.png": 2, "1.webp": 3}
                image = next((p for p in sorted(candidate.parent.iterdir(),
                              key=lambda p: (priority.get(p.name.casefold(), 9), p.name))
                              if p.is_file() and p.name.casefold() in {"1.jpg", "1.jpeg", "1.png", "1.webp"}), None)
        if image is None and item["image"] and not re.match(r"^[a-z]+://", item["image"], re.I):
            try:
                candidate = resolve_image(image_root, item["image"].replace("\\", "/"))
                image = candidate if candidate.is_file() else None
            except ValueError:
                pass
        item["image"] = image.resolve().relative_to(image_root.resolve()).as_posix() if image else ""
        if not image:
            missing_images.append(item["product_id"])
        converted.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(Product.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(converted)
        products = load_catalog(output, image_root)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    report = {"products": len(products), "column_mapping": selected,
              "matched_cover_images": len(products)-len(missing_images),
              "products_without_images": missing_images, "groups_defaulted_to_product_id": inferred_groups,
              "notes": ["No source file or image was modified.",
                        "Review product family IDs before splitting relevance judgments.",
                        "Only the cover image is indexed. Missing images leave text-only products."]}
    output.with_suffix(".import.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
