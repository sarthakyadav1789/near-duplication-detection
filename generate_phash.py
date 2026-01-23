import os
import csv
from PIL import Image
import imagehash
from tqdm import tqdm

# ================= CONFIG =================
ROOT_DIR = "E:/IMAGES/0/0"
FOLDERS = [str(i) for i in range(9)]   # 0 to 8
OUTPUT_CSV = "image_phashes.csv"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =========================================

def compute_phash(image_path):
    """Compute pHash for a single image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return str(imagehash.phash(img))  # hex string

def main():
    rows = []

    for folder in FOLDERS:
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(VALID_EXTS)
        ]

        for img_name in tqdm(images, desc=f"Processing folder {folder}"):
            img_path = os.path.join(folder_path, img_name)
            try:
                phash = compute_phash(img_path)
                rows.append([
                    folder,
                    img_name,
                    img_path,
                    phash
                ])
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "filename", "full_path", "phash"])
        writer.writerows(rows)

    print(f"\n✅ Done! Saved {len(rows)} hashes to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
