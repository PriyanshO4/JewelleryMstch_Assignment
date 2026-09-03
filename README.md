# Jewellery Matcher

A prototype that recommends matching earrings for a given necklace based on visual similarity.

## Problem

Given a small inventory of 5 necklaces and 15 earrings, recommend the most visually similar earrings for a selected necklace.

## Approach

Training a model from scratch doesn't make sense with only 20 images. Instead, I used a **pretrained CLIP model** (`openai/clip-vit-base-patch32`) to generate image embeddings and ranked earrings using **cosine similarity**.

### How the matching works

```
Necklace Image → CLIP Embedding (512-dim vector)
                        ↓
              Cosine Similarity vs. 15 precomputed earring embeddings
                        ↓
              Ranked earrings (highest similarity first)
```

1. Each earring image is passed through CLIP's image encoder to get a 512-dimensional embedding vector. These are computed once at startup and cached.
2. When the user selects a necklace, only that image gets embedded.
3. The necklace embedding is compared against all 15 earring embeddings using cosine similarity (dot product of L2-normalized vectors).
4. Earrings are sorted by score and the top matches are displayed.

No model training was performed — the CLIP model is pretrained and used as-is.

## Technologies Used

- **Python 3.9+**
- **PyTorch** — tensor ops and model inference
- **Hugging Face Transformers** — loading the pretrained CLIP model
- **Streamlit** — web UI
- **Pillow** — image loading
- **pandas** — CSV parsing
- **NumPy** — embedding math (normalization, dot product)

## Project Structure

```
jewellery-matcher/
├── app.py                  # Streamlit UI
├── matcher.py              # Embedding + matching logic
├── requirements.txt        # Dependencies
├── candidate_dataset.csv   # Product inventory
├── images/                 # Necklace and earring images
├── README.md
├── DEMO_GUIDE.md
└── .gitignore
```

## Installation & Running

```bash
cd jewellery-matcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the CLIP model (~600 MB), cached after that.

## Limitations

- CLIP compares images visually — it doesn't understand jewellery-specific attributes like metal type or gemstones.
- The model is trained on general images, not jewellery. A fine-tuned model would give better results.
- Small inventory (15 earrings) limits recommendation diversity.

## Possible Improvements

- Fine-tune CLIP on a larger jewellery dataset.
- Add text-based filtering (metal, colour) alongside visual similarity.
- Use CLIP's text encoder for natural language queries ("gold earrings with pearls").
- Scale with FAISS for approximate nearest neighbour search on larger inventories.
