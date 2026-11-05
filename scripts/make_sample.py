"""Build fictional catalog fixtures. These images are not a model benchmark."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "data/sample"


def main():
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    families = [
        ("jar", "Glass storage jar", "برطمان تخزين زجاجي", "Kitchen", "transparent glass kitchen container with lid", "وعاء مطبخ زجاجي شفاف مع غطاء", "#a8c6b6"),
        ("lamp", "Table lamp", "مصباح طاولة", "Lighting", "fabric shade bedside reading light", "اضاءة جانبية للقراءة بجانب السرير", "#c7b08b"),
        ("bag", "Canvas backpack", "حقيبة ظهر قماشية", "Bags", "school travel bag with shoulder straps", "شنطة للسفر والمدرسة مع احزمة كتف", "#849787"),
        ("weights", "Hex dumbbell", "دمبل سداسي", "Fitness", "strength training home gym exercise weight", "اوزان تمارين القوة في المنزل", "#60646b"),
        ("vase", "Ceramic flower vase", "مزهرية سيراميك", "Decor", "decorative flower holder for living room table", "حامل زهور لتزيين طاولة غرفة المعيشة", "#b58b73"),
        ("mug", "Ceramic coffee mug", "كوب قهوة سيراميك", "Kitchen", "drinking cup with handle for coffee and tea", "كوب بمقبض لشرب القهوة والشاي", "#9cafb4"),
    ]
    rows, queries = [], []
    for index, (family, en, ar, category, description, description_ar, color) in enumerate(families):
        ids = []
        for variant, size in enumerate(["Small", "Large"]):
            pid = f"{family}-{variant+1}"
            ids.append(pid)
            image = Image.new("RGB", (320, 320), "#f5f2eb")
            draw = ImageDraw.Draw(image)
            bounds = (80 - variant * 12, 65, 240 + variant * 12, 250)
            if family == "lamp":
                draw.polygon([(105, 65), (215, 65), (260, 170), (60, 170)], fill=color)
                draw.rectangle((155, 170, 165, 260), fill="#807464")
                draw.ellipse((110, 250, 210, 270), fill="#807464")
            elif family == "weights":
                draw.rectangle((65, 140, 255, 175), fill="#a4a4a4")
                draw.rounded_rectangle((45, 100, 115, 215), radius=12, fill=color)
                draw.rounded_rectangle((205, 100, 275, 215), radius=12, fill=color)
            elif family == "vase":
                draw.ellipse((80, 120, 240, 265), fill=color)
                draw.rectangle((130, 65, 190, 170), fill=color)
            else:
                draw.rounded_rectangle(bounds, radius=25, fill=color, outline="#727a71", width=3)
                if family == "jar":
                    draw.rounded_rectangle((75-variant*12, 55, 245+variant*12, 90), radius=9, fill="#aa9376")
                if family == "bag":
                    draw.rounded_rectangle((110, 165, 210, 235), radius=12, outline="#596959", width=4)
                if family == "mug":
                    draw.arc((220, 105, 285, 215), -90, 90, fill=color, width=17)
            draw.text((12, 294), f"{family} / {size} / synthetic fixture", fill="#555555")
            image.save(ROOT / f"images/{pid}.jpg", quality=90)
            rows.append([pid, family, f"001000{index}{variant}", f"{size} {en}", f"{ar} {'صغير' if variant == 0 else 'كبير'}", description, description_ar, category, 20+index*12+variant*5, 0 if pid == 'lamp-2' else 15, f"images/{pid}.jpg"])
        split = ["train", "dev", "test"][index//2]
        for language, text in [("en", en), ("ar", ar)]:
            queries.append({"query_id": f"{family}-{language}", "text": text,
                            "language": language, "modality": "text", "split": split,
                            "relevance": {pid: 2 for pid in ids}})
        # Keep image queries separate from text queries when reporting metrics.
        view = Image.open(ROOT / f"images/{ids[0]}.jpg").rotate(5, fillcolor="#f5f2eb")
        view.save(ROOT / f"images/{family}-query.jpg", quality=85)
        queries.append({"query_id": f"{family}-image", "image": f"images/{family}-query.jpg",
                        "language": "none", "modality": "image", "split": split,
                        "relevance": {pid: 2 for pid in ids}})
    with (ROOT / "catalog.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["product_id", "group_id", "barcode", "name_en", "name_ar", "description_en", "description_ar", "category", "price", "stock", "image"])
        writer.writerows(rows)
    for query in queries:
        query["relevance"] = {row[0]: query["relevance"].get(row[0], 0) for row in rows}
    (ROOT / "queries.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
