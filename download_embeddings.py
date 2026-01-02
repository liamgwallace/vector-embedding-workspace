#!/usr/bin/env python3
"""
Download GloVe embeddings for the Vector Embedding Playground.
"""

import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
EMBEDDINGS_DIR = "embeddings"
TARGET_FILE = "glove.6B.50d.txt"


def download_progress(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()


def main():
    # Create embeddings directory
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    target_path = os.path.join(EMBEDDINGS_DIR, TARGET_FILE)

    # Check if already exists
    if os.path.exists(target_path):
        print(f"Embeddings already exist at {target_path}")
        return

    zip_path = os.path.join(EMBEDDINGS_DIR, "glove.6B.zip")

    # Download if zip doesn't exist
    if not os.path.exists(zip_path):
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
    print(f"\nExtracting {TARGET_FILE}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(TARGET_FILE, EMBEDDINGS_DIR)
        print(f"Extracted to {target_path}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)

    # Optionally remove the zip to save space
    print(f"\nCleaning up zip file...")
    try:
        os.remove(zip_path)
        print("Done!")
    except:
        print(f"Could not remove {zip_path}, you can delete it manually to save space")

    print(f"\n✓ Setup complete! You can now run: python server.py")


if __name__ == "__main__":
    main()
