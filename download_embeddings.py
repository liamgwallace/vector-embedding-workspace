#!/usr/bin/env python3
"""
Download FastText embeddings for the Vector Embedding Playground.
Downloads the crawl-300d-2M-subword model which supports OOV (out-of-vocabulary) words
through character n-gram subword vectors.
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
from sklearn.decomposition import PCA

# FastText crawl-300d-2M-subword model (supports OOV via subwords)
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
EMBEDDINGS_DIR = "embeddings"
MODEL_FILE = "crawl-300d-2M-subword.bin"
PROCESSED_FILE = "fasttext.crawl-300d-2M.npz"
PCA_FILE = "fasttext.crawl-300d-2M.2d.npz"

# Number of most frequent words to pre-compute for fast similarity search
VOCAB_SIZE = 500000


def download_progress(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    downloaded_mb = count * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    sys.stdout.write(f"\rDownloading: {percent}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)")
    sys.stdout.flush()


def load_common_words(model, target_size=500000):
    """
    Workaround for get_words() Unicode errors: Extract words by downloading
    a large word frequency list and filtering for words with valid vectors.
    """
    print("  Downloading large word frequency list...")

    words = []

    # Try to download a comprehensive word list (330K+ words)
    try:
        import urllib.request
        import gzip

        # Try getting words from multiple sources
        sources = [
            # SCOWL word list (comprehensive English word list)
            "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        ]

        word_list = []
        for url in sources:
            try:
                print(f"  Downloading from {url.split('/')[-1]}...")
                response = urllib.request.urlopen(url, timeout=30)
                content = response.read().decode('utf-8')
                word_list.extend(content.strip().split('\n'))
                print(f"  Downloaded {len(word_list)} words")
                break
            except:
                continue

        if not word_list:
            raise Exception("Could not download word list from any source")

        # Test each word to see if the model knows it
        print(f"  Testing words with model (targeting {target_size} words)...")
        valid_count = 0

        for i, word in enumerate(word_list):
            if (i + 1) % 10000 == 0:
                print(f"    Tested {i + 1} words, found {valid_count} valid...")

            word = word.strip().lower()
            if word and len(word) > 1 and word.isalpha():
                try:
                    # Test if we can get a vector for this word
                    vec = model.get_word_vector(word)
                    if vec is not None and len(vec) > 0:
                        words.append(word)
                        valid_count += 1
                except:
                    pass

            if len(words) >= target_size:
                break

        print(f"  Successfully extracted {len(words)} valid words from model")
        return words

    except Exception as e:
        print(f"  Error: Could not complete word extraction: {e}")
        print(f"  The FastText model file may be corrupted.")
        print(f"  Please delete the embeddings folder and run this script again.")
        raise


def process_fasttext_model(model_path, npz_path, vocab_size=VOCAB_SIZE):
    """
    Load FastText binary model and extract vocabulary embeddings.
    Pre-computes normalized vectors for fast similarity search.
    """
    import fasttext

    print(f"\nLoading FastText model from {model_path}...")
    print("This may take a minute...")

    model = fasttext.load_model(model_path)

    # Get all words from the model
    # Try get_words() first, fall back to alternatives if it fails
    try:
        all_words = model.get_words()
    except (UnicodeDecodeError, Exception) as e:
        print(f"  Warning: get_words() failed with {type(e).__name__}, using workaround...")
        # Workaround: Try words attribute, but it may also fail
        try:
            if hasattr(model, 'words'):
                all_words = model.words
            else:
                raise AttributeError("No words attribute")
        except (UnicodeDecodeError, AttributeError, Exception):
            # Alternative: manually extract common words and test them
            print("  Extracting vocabulary by testing common words...")
            # Load a common word list
            all_words = load_common_words(model)

    print(f"  Model vocabulary size: {len(all_words)} words")

    # Limit to top N words (they're already sorted by frequency)
    words_to_process = all_words[:vocab_size]
    print(f"  Processing top {len(words_to_process)} words...")

    words = []
    vectors = []

    for i, word in enumerate(words_to_process):
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1} words...")

        vector = model.get_word_vector(word).astype(np.float32)

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

    return embedding_matrix, words


def compute_pca_2d(embedding_matrix, words, pca_path):
    """
    Compute PCA to reduce embeddings to 2D for visualization.
    Saves both the 2D coordinates and the PCA transformation parameters.
    """
    print(f"\nComputing PCA reduction to 2D...")
    print(f"  Input shape: {embedding_matrix.shape}")

    # Fit PCA on the full embedding matrix
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embedding_matrix)

    print(f"  Output shape: {embeddings_2d.shape}")
    print(f"  Explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    print(f"  Saving to {pca_path}...")

    # Save 2D embeddings along with PCA transformation parameters
    np.savez_compressed(pca_path,
                       embeddings_2d=embeddings_2d.astype(np.float32),
                       words=words,
                       pca_mean=pca.mean_.astype(np.float32),
                       pca_components=pca.components_.astype(np.float32))

    print(f"  Saved! File size: {os.path.getsize(pca_path) / 1024 / 1024:.1f} MB")


def main():
    # Create embeddings directory
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    model_path = os.path.join(EMBEDDINGS_DIR, MODEL_FILE)
    processed_path = os.path.join(EMBEDDINGS_DIR, PROCESSED_FILE)
    pca_path = os.path.join(EMBEDDINGS_DIR, PCA_FILE)

    # Check if both processed files already exist
    if os.path.exists(processed_path) and os.path.exists(pca_path) and os.path.exists(model_path):
        print(f"FastText model exists at {model_path}")
        print(f"Processed embeddings already exist at {processed_path}")
        print(f"PCA 2D embeddings already exist at {pca_path}")
        return

    zip_path = os.path.join(EMBEDDINGS_DIR, "crawl-300d-2M-subword.zip")

    # Download if zip doesn't exist and model doesn't exist
    if not os.path.exists(zip_path) and not os.path.exists(model_path):
        print(f"Downloading FastText crawl-300d-2M-subword model...")
        print(f"URL: {FASTTEXT_URL}")
        print(f"This is about 2GB and may take several minutes.\n")

        try:
            urllib.request.urlretrieve(FASTTEXT_URL, zip_path, download_progress)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  {FASTTEXT_URL}")
            print(f"  Then extract {MODEL_FILE} to the {EMBEDDINGS_DIR}/ folder")
            sys.exit(1)

    # Extract the .bin file
    if not os.path.exists(model_path):
        print(f"\nExtracting {MODEL_FILE}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extract(MODEL_FILE, EMBEDDINGS_DIR)
            print(f"Extracted to {model_path}")
        except Exception as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)

    # Process embeddings (extract vocabulary, normalize and convert to .npz)
    embedding_matrix, words = process_fasttext_model(model_path, processed_path)

    # Compute PCA 2D reduction for visualization
    compute_pca_2d(embedding_matrix, words, pca_path)

    # Clean up zip file to save space (keep the .bin for OOV support)
    print(f"\nCleaning up temporary files...")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  Removed {zip_path}")
    except:
        print(f"  Could not remove {zip_path}")

    print(f"\n" + "=" * 60)
    print("Setup complete!")
    print(f"  - FastText model: {model_path} (kept for OOV support)")
    print(f"  - Pre-computed vectors: {processed_path}")
    print(f"  - 2D visualization: {pca_path}")
    print(f"\nYou can now run: python server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
