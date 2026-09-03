# Demo Recording Guide

Record a 1–2 minute screen recording showing the following steps.

## Before Recording

Make sure the app is running:

```bash
streamlit run app.py
```

## What to Show

1. **App startup** — show the app loaded in the browser. Briefly point out the sidebar with the necklace dropdown.

2. **Select a necklace** — pick one from the dropdown (e.g., N01). The necklace image appears in the sidebar.

3. **Run the matching** — click **Find Matching Earrings**. Wait a second for the results.

4. **Show the results** — the top earrings are displayed with their IDs and similarity scores. Mention that higher scores mean more visual similarity.

5. **Try another necklace** — change the selection to a different necklace (e.g., N03), click the button again, and show that the recommendations change.

6. **Brief explanation** (voiceover or text overlay):
   - "Each image is converted into an embedding using a pretrained CLIP model."
   - "The necklace embedding is compared against all earring embeddings using cosine similarity."
   - "Earrings are ranked by similarity and the top matches are shown."
   - "No model training was done — the model is pretrained and used as-is."

## Tips

- Keep it under 2 minutes.
- You don't need to show the code — focus on the working app and the results.
- Speak naturally. It's a prototype demo, not a product launch.
