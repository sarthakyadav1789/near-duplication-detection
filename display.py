from PIL import Image
import matplotlib.pyplot as plt
import os

def show_images(image_paths, similarities=None, cols=3):
    if not image_paths:
        print("No images provided")
        return

    
    if similarities is not None and len(similarities) != len(image_paths):
        raise ValueError("Length of similarities must match image_paths")

    # Filtering valid paths
    valid_paths = []
    valid_scores = []

    for i, p in enumerate(image_paths):
        if os.path.exists(p):
            valid_paths.append(p)
            if similarities is not None:
                valid_scores.append(similarities[i])
        else:
            print(f"❌ File not found: {p}")

    if not valid_paths:
        print("No valid images to display")
        return

    n = len(valid_paths)
    rows = (n + cols - 1) // cols


    

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, img_path in enumerate(valid_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
            axes[i].axis("off")

           
            if similarities is not None:
                title = f"{os.path.basename(img_path)}\nSim: {valid_scores[i]:.4f}"
            else:
                title = os.path.basename(img_path)

            axes[i].set_title(title, fontsize=10)

        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            axes[i].axis("off")

    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
