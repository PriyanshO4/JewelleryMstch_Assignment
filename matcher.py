import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# using clip-vit-base-patch32 — lightweight and good enough for visual similarity
MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    """Pick GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device=None):
    """Load CLIP model and processor. Call once and cache."""
    if device is None:
        device = get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model, processor, device


def load_dataset(csv_path, image_dir):
    """Read the CSV and split into necklaces / earrings.
    Skips rows whose image file is missing on disk."""
    df = pd.read_csv(csv_path)

    # validate images exist
    valid = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_file"])
        if os.path.isfile(img_path):
            valid.append(row)
        else:
            print(f"Warning: missing image {row['image_file']}, skipping")
    df = pd.DataFrame(valid)

    necklaces = df[df["product_type"] == "Necklace"].reset_index(drop=True)
    earrings = df[df["product_type"] == "Earrings"].reset_index(drop=True)
    return necklaces, earrings


def get_image_embedding(image, model, processor, device):
    """Generate a normalized CLIP embedding for a single PIL image."""
    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.get_image_features(**inputs)
        # newer transformers may return a structured output instead of a tensor
        if not isinstance(output, torch.Tensor):
            features = output.pooler_output if hasattr(output, 'pooler_output') else output[0]
        else:
            features = output
        # L2 normalize so dot product == cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def precompute_earring_embeddings(earrings_df, image_dir, model, processor, device):
    """Compute embeddings for all earring images. Returns a numpy matrix (N x dim)
    and a list of ids in the same order."""
    embeddings = []
    ids = []
    for _, row in earrings_df.iterrows():
        img_path = os.path.join(image_dir, row["image_file"])
        try:
            img = Image.open(img_path).convert("RGB")
            emb = get_image_embedding(img, model, processor, device)
            embeddings.append(emb)
            ids.append(row["id"])
        except Exception as e:
            print(f"Could not process {row['image_file']}: {e}")
    return np.array(embeddings), ids


def find_matching_earrings(necklace_embedding, earring_embeddings, earring_ids, top_k=3):
    """Rank earrings by cosine similarity to the necklace embedding.
    Returns list of (id, score) sorted high to low."""
    # both are already L2-normalized, so dot product = cosine sim
    scores = earring_embeddings @ necklace_embedding
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    results = [(earring_ids[i], float(scores[i])) for i in ranked_idx]
    return results
