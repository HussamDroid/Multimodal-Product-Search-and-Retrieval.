# Import an existing product master

The importer accepts `.xlsx`, UTF-8 `.csv` and `.tsv`. It maps common export headings, removes HTML from searchable text and matches the cover image in each barcode folder. It never edits the source spreadsheet or photos.

## Windows example

From `D:\souqlens`, install Excel support:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[import]"
```

Place a copy of the product export under `D:\souqlens\data\private\master.xlsx`. Keep the image folder in its existing location. Replace the image root below with the actual path:

```powershell
.\.venv\Scripts\souqlens.exe prepare-catalog --source data/private/master.xlsx --image-root "Z:\Mini Mall\Mini-Mall-Images" --output data/private/catalog-v1.csv
```

For Black Rose, use its actual product-image parent folder instead. This example does not assume the mapped drive is available on a hosted server.

The importer produces `catalog-v1.csv` and `catalog-v1.import.json`. Review the matched columns, number of images found and list of products without images. Missing covers do not remove products; those products can use text embeddings only.

Common aliases include `Model Number` → `product_id`, `Barcode`/`EAN` → `barcode`, `Product Name` → `name_en`, `Description` → `description_en` and `Quantity` → `stock`. Arabic names/descriptions and categories are mapped when present.

## Explicit mapping

If the source headings differ or auto-mapping is ambiguous, create a local mapping JSON:

```json
{
  "product_id": "Model Number",
  "barcode": "Barcode",
  "name_en": "English Product Title",
  "name_ar": "Arabic Product Title",
  "description_en": "English Description",
  "price": "Retail Price",
  "stock": "Warehouse Quantity"
}
```

Pass it with `--mapping data/private/column-map.json`. Use `--sheet "Products"` to select an Excel worksheet. Missing required fields and invalid numbers stop the import; rows are not silently discarded.

## Important data details

- Barcodes must retain their complete text. Scientific notation is rejected. Simple Excel zero-padding formats are preserved; digits already lost by Excel cannot be recovered by the importer.
- If no family column exists, each product gets its own ID as `group_id`. This is a placeholder, not automatic variant detection. Review families before preparing train/dev/test judgments.
- Images are matched from `barcode/1.jpg`, with JPEG/PNG/WebP cover variants also accepted. Only the cover is indexed initially. RAW folders are not scanned.
- Existing web URLs are not downloaded. Provide the local image root or a separate image archive to prepare a hosted bundle.
- The importer only exports canonical product fields. It does not copy customer information or unrelated spreadsheet columns into the catalog.
- `data/private/` and generated model/index artifacts are ignored by Git. Keep real exports and generated catalogs out of the public source repository.

After reviewing the report:

```powershell
.\.venv\Scripts\souqlens.exe index --catalog data/private/catalog-v1.csv --image-root "Z:\Mini Mall\Mini-Mall-Images" --output artifacts/real-index-v1
.\.venv\Scripts\souqlens.exe serve --index artifacts/real-index-v1 --image-root "Z:\Mini Mall\Mini-Mall-Images"
```

Use a new output filename/index directory for later versions. Real relevance judgments are a separate step; imported descriptions are not automatic evaluation labels.

## OpenCart masters with repeated headers and barcodes

For exports containing an OpenCart field block followed by warehouse helper columns, use:

```powershell
.\.venv\Scripts\souqlens.exe prepare-opencart --source data/private/MASTER-BR.xlsx --image-root data/private/blackrose-images --output data/private/blackrose-v1.csv --duplicates last
```

This command includes only enabled products with supplied cover images. It reads the first Barcode column from the OpenCart block and uses barcodes as product IDs, so variants sharing a model number stay separate. Model numbers provide provisional family groups; review them before evaluation. Only English export rows are accepted by this adapter.

Duplicate handling must be explicit: `first`, `last`, or `error`. For this Black Rose demo, the owner selected `last`. Row order is a selection policy, not evidence of update time. The import report records selected source rows and conflicting values. Empty fields in the selected row stay empty; the command does not combine descriptions or categories from other rows. The original sheet and photos are unchanged.

Keep the generated catalog, conflict report and images private. Upload a checksummed bundle to a private Hugging Face dataset; source deployment alone does not attach catalog data to the Space.
