import os
import zipfile
import urllib.request

import numpy as np
import torch
import umap
from wordfreq import top_n_list

print("...")
# ============================================================
# CONFIG
# ============================================================

TOP_N_WORDS = 100
GLOVE_DIM = 50
UMAP_DIM = 4

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_TXT = f"glove.6B.{GLOVE_DIM}d.txt"

OUTPUT_FILE = "compressed_word_embeddings.pt"

# ============================================================
# DOWNLOAD GLOVE IF NEEDED
# ============================================================

if not os.path.exists(GLOVE_TXT):

    if not os.path.exists(GLOVE_ZIP):
        print("Downloading GloVe...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)

    print("Extracting GloVe...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as zip_ref:
        zip_ref.extract(GLOVE_TXT)

# ============================================================
# GET TOP WORDS
# ============================================================

print("Getting top words from wordfreq...")

words = top_n_list(
    lang="en",
    n=TOP_N_WORDS
)

print(words)

# ============================================================
# LOAD ONLY NEEDED GLOVE EMBEDDINGS
# ============================================================

print("Loading GloVe embeddings...")

word_set = set(words)

original_embeddings = {}

with open(GLOVE_TXT, "r", encoding="utf-8") as f:

    for line in f:

        split = line.strip().split()

        word = split[0]

        if word in word_set:

            vector = np.array(
                split[1:],
                dtype=np.float32
            )

            original_embeddings[word] = vector

# ============================================================
# FILTER WORDS THAT EXIST IN GLOVE
# ============================================================

valid_words = [
    w for w in words
    if w in original_embeddings
]

embedding_matrix = np.stack([
    original_embeddings[w]
    for w in valid_words
])

print(f"Found embeddings for {len(valid_words)} words")

# ============================================================
# UMAP DIMENSIONALITY REDUCTION
# ============================================================

print("Running UMAP compression...")

reducer = umap.UMAP(
    n_components=UMAP_DIM,
    random_state=42
)

compressed_embeddings = reducer.fit_transform(
    embedding_matrix
)

# ============================================================
# BUILD MAPS
# ============================================================

word_to_embedding = {}
embedding_to_word = {}

for word, emb in zip(valid_words, compressed_embeddings):

    emb_tensor = torch.tensor(
        emb,
        dtype=torch.float32
    )

    word_to_embedding[word] = emb_tensor

    # tuple key for reverse lookup
    embedding_to_word[tuple(emb.tolist())] = word

# ============================================================
# SAVE PT FILE
# ============================================================

save_dict = {
    "word_to_embedding": word_to_embedding,
    "embedding_to_word": embedding_to_word,
    "metadata": {
        "top_n_words": TOP_N_WORDS,
        "glove_dim": GLOVE_DIM,
        "compressed_dim": UMAP_DIM
    }
}

torch.save(save_dict, OUTPUT_FILE)

print(f"\nSaved to: {OUTPUT_FILE}")

# ============================================================
# EXAMPLE USAGE
# ============================================================

loaded = torch.load(OUTPUT_FILE)

w2e = loaded["word_to_embedding"]

print("\nExample:")

example_word = valid_words[0]

print("Word:", example_word)
print("Embedding:", w2e[example_word])