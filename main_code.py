# main.py
import csv
from PIL import Image
import imagehash
import os

CSV_FILE = "E:/projects/near-duplicate-detection/image_phashes.csv"
IMAGE_BASE = "E:/IMAGES/0/0"
HAMMING_DUPLICATE = 7  


def compute_phash(image_path):
    #Compute pHash of an image
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return imagehash.phash(img)

def find_similar_images(input_image_path):
    
    #pHash-based filtering.
   
   
    input_hash = compute_phash(input_image_path)
    query_name = os.path.basename(input_image_path)

    direct_duplicates = []
    ai_candidates = []

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["filename"] == query_name:
                continue
            stored_hash = imagehash.hex_to_hash(row["phash"])
            dist = input_hash - stored_hash
            record = (row["folder"], row["filename"], dist)
            if dist <= HAMMING_DUPLICATE:
                direct_duplicates.append(record)
            else:
                ai_candidates.append(record)

    return direct_duplicates, ai_candidates

def get_paths_and_scores(matches):
    paths = []
    scores = []
    for folder, filename, dist in matches:
        img_path = os.path.join(IMAGE_BASE, str(folder), filename)
        if os.path.exists(img_path):
            paths.append(img_path)
            scores.append(1 - dist / 64)
    return paths, scores

def process_image(input_image_path):
    
    #Main processing function.
    """
        direct_paths: paths of direct pHash duplicates
        direct_scores: similarity scores
        ai_candidates_paths: paths for CLIP/PCA processing
        ai_candidates_files: filenames only for filtering"""
    
    direct, candidates = find_similar_images(input_image_path)
    direct_paths, direct_scores = get_paths_and_scores(direct)
    ai_candidates_paths = [os.path.join(IMAGE_BASE, str(f), fn) for f, fn, _ in candidates]
    ai_candidates_files = [fn for _, fn, _ in candidates]
    return direct_paths, direct_scores, ai_candidates_paths, ai_candidates_files
