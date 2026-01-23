import os
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import pandas as pd


DATASET_ROOT = "E:/IMAGES/0/0"             
OUTPUT_CSV = "clip_embeddings.csv"      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "E:/models/open_clip_model.safetensors"  
MODEL_NAME = "ViT-B-32"  
BATCH_SIZE = 32           

# Loading Model
print("Loading OpenCLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=MODEL_PATH
)
model = model.to(DEVICE)
model.eval()

#Storing image paths
image_paths = []
for folder_name in sorted(os.listdir(DATASET_ROOT)):
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if os.path.isdir(folder_path):
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".jpg"):
                image_paths.append({
                    "folder": folder_name,
                    "file": fname,
                    "path": os.path.join(folder_path, fname)
                })

print(f"Found {len(image_paths)} images.")

#  Generating Image Embedding
embeddings_list = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches"):
    batch = image_paths[i:i+BATCH_SIZE]
    
    # Loading and preprocessing images
    imgs = []
    for img_info in batch:
        try:
            img = Image.open(img_info["path"]).convert("RGB")
            imgs.append(preprocess(img))
        except Exception as e:
            print(f"Skipping {img_info['path']} due to error: {e}")

    if not imgs:
        continue

    imgs_tensor = torch.stack(imgs).to(DEVICE)

    # Encoding images
    with torch.no_grad():
        batch_embeddings = model.encode_image(imgs_tensor)
        batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True) 

    batch_embeddings = batch_embeddings.cpu().numpy()

   
    for j, emb in enumerate(batch_embeddings):
        emb_dict = {
            "folder": batch[j]["folder"],
            "file": batch[j]["file"],
        }
        emb_dict.update({f"dim_{k}": float(emb[k]) for k in range(emb.shape[0])})
        embeddings_list.append(emb_dict)

# ================= SAVE TO CSV =================
df = pd.DataFrame(embeddings_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved embeddings to {OUTPUT_CSV}")
