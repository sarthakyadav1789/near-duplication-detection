import torch
import open_clip
from PIL import Image
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "E:/models/open_clip_model.safetensors"
MODEL_NAME = "ViT-B-32"

# Loading Model
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=MODEL_PATH
)
model = model.to(DEVICE)
model.eval()

#embeddings
def get_image_embedding(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Loading & preprocessing image
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)  

    # Encoding and normalization
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze(0).cpu() 
