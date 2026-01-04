#!/usr/bin/env python3
"""
Download GloVe embeddings for the Vector Embedding Playground.
Pre-processes embeddings by normalizing vectors and saving in optimized .npz format.
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
EMBEDDINGS_DIR = "embeddings"
TARGET_FILE = "glove.6B.50d.txt"
PROCESSED_FILE = "glove.6B.50d.npz"


def download_progress(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()


def process_embeddings(txt_path, npz_path):
    """
    Load GloVe text file, normalize vectors, and save as optimized .npz format.
    """
    print(f"\nProcessing embeddings from {txt_path}...")

    words = []
    vectors = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i + 1) % 50000 == 0:
                print(f"  Processed {i + 1} words...")

            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            # Normalize the vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            words.append(word)
            vectors.append(vector)

    # Convert to numpy arrays
    embedding_matrix = np.array(vectors, dtype=np.float32)

    print(f"  Total: {len(words)} words with {embedding_matrix.shape[1]} dimensions")
    print(f"  Saving to {npz_path}...")

    # Save as compressed numpy format
    np.savez_compressed(npz_path,
                       embeddings=embedding_matrix,
                       words=np.array(words))

    print(f"  Saved! File size: {os.path.getsize(npz_path) / 1024 / 1024:.1f} MB")


def main():
    # Create embeddings directory
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    target_path = os.path.join(EMBEDDINGS_DIR, TARGET_FILE)
    processed_path = os.path.join(EMBEDDINGS_DIR, PROCESSED_FILE)

    # Check if processed file already exists
    if os.path.exists(processed_path):
        print(f"Processed embeddings already exist at {processed_path}")
        return

    zip_path = os.path.join(EMBEDDINGS_DIR, "glove.6B.zip")

    # Download if zip doesn't exist
    if not os.path.exists(zip_path) and not os.path.exists(target_path):
        print(f"Downloading GloVe embeddings from Stanford NLP...")
        print(f"URL: {GLOVE_URL}")
        print(f"This is about 862MB and may take a few minutes.\n")

        try:
            urllib.request.urlretrieve(GLOVE_URL, zip_path, download_progress)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  {GLOVE_URL}")
            print(f"  Then extract {TARGET_FILE} to the {EMBEDDINGS_DIR}/ folder")
            sys.exit(1)

    # Extract just the 50d file
    if not os.path.exists(target_path):
        print(f"\nExtracting {TARGET_FILE}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extract(TARGET_FILE, EMBEDDINGS_DIR)
            print(f"Extracted to {target_path}")
        except Exception as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)

    # Process embeddings (normalize and convert to .npz)
    process_embeddings(target_path, processed_path)

    # Clean up text file and zip to save space
    print(f"\nCleaning up temporary files...")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  Removed {zip_path}")
    except:
        print(f"  Could not remove {zip_path}")

    try:
        if os.path.exists(target_path):
            os.remove(target_path)
            print(f"  Removed {target_path} (keeping only optimized .npz)")
    except:
        print(f"  Could not remove {target_path}")

    print(f"\nSetup complete! You can now run: python server.py")


if __name__ == "__main__":
    main()
