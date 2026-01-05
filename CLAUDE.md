# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vector Embedding Playground is a Flask-based web application for exploring word vector embeddings through two main tools:

1. **Word Equations** (`/` route): Direct vector arithmetic like `king - man + woman ≈ queen`
2. **Semantic Concepts** (`/transform` route): Pattern learning from word pair examples (e.g., "animal → food" transformation)

The application uses FastText crawl-300d-2M-subword embeddings (500K words, 300 dimensions) with OOV (out-of-vocabulary) support via character n-gram subword vectors.

## Setup & Running

```bash
# Install dependencies (Windows note: uses fasttext-wheel instead of fasttext)
pip install -r requirements.txt

# Download FastText embeddings (~2GB download, extracts to ~7GB)
python download_embeddings.py

# Start the development server
python server.py
```

The server runs at http://localhost:5000 with Flask debug mode enabled.

## Architecture

### Core Data Structures (server.py)

The server maintains global state for fast similarity search:

- `embedding_matrix`: Normalized numpy array (500K × 300) of pre-computed FastText vectors. Pre-normalization allows cosine similarity via dot product.
- `words_list`: Vocabulary array corresponding to embedding matrix rows.
- `word_to_idx`: Dictionary for O(1) word → index lookup.
- `fasttext_model`: Full FastText binary model (~7GB) loaded for OOV word handling.
- `embeddings_2d`: Pre-computed 2D PCA projections for visualization.
- `pca_mean` / `pca_components`: PCA transformation parameters for projecting arbitrary vectors to 2D.
- `current_transformation`: Global state storing the learned transformation vector from word pairs (used in `/transform` tool).

### Vector Arithmetic Engine (server.py:229-301)

`parse_and_evaluate(equation, registers)` is the core equation parser that:

1. **Supports scalar multiplication and parentheses**: `0.8 * (woman - man) + king`
2. **Extracts word tokens** via regex: `\b[a-zA-Z]+\b`
3. **Resolves vectors** for each word using `get_vector()`:
   - First checks pre-computed vocabulary (fast)
   - Falls back to FastText model for OOV words (~5-10ms per word)
4. **Uses Python's `eval()`** in a restricted namespace containing only word vectors and numpy
5. **Returns**: result vector, processed equation, words used, error message, and OOV words list

### OOV Word Handling (server.py:83-106)

`get_vector(word)` implements two-tier lookup:

1. **Pre-computed vocabulary** (500K most frequent words): O(1) dictionary lookup
2. **FastText subword vectors**: For typos, rare words, and novel terms. The model generates vectors on-the-fly using character n-grams (e.g., "3-6 character substrings").

OOV words can be used in equations but won't appear in similarity search results (only vocabulary words are searched).

### Similarity Search (server.py:144-176)

`find_nearest_neighbors(vector, n, exclude_words)` performs:

1. **Vectorized cosine similarity**: Matrix-vector dot product across all 500K words
2. **Efficient top-k selection**: Uses `np.argpartition()` which is O(n) instead of O(n log n) for full sort
3. **Word exclusion**: Sets similarities to `-np.inf` to exclude input words from results

### Pattern Learning (server.py:179-226)

`calculate_average_transformation(pairs)` implements robust transformation learning:

1. **Calculates transformation vectors**: `vec_to - vec_from` for each word pair
2. **Robust averaging with outlier detection**:
   - Computes pairwise cosine similarities between all transformation vectors
   - Removes outliers (transformations with < 80% of median similarity)
   - Averages remaining "inlier" transformations
3. **Returns**: averaged transformation, valid pairs, outlier count, and outlier list

This allows learning concepts like "animal → food" from examples and applying them to new words.

### API Endpoints

**Word Equations Tool:**
- `POST /api/calculate`: Execute vector equation, return top-n neighbors with 2D coords
- `GET /api/check_word/<word>`: Validate word (returns `exists`, `oov_supported`, `can_use`)
- `GET /api/vocab_size`: Return vocabulary stats and OOV support status
- `GET /api/vocab_sample`: Sample ~3000 words with 2D coords for background visualization

**Semantic Concepts Tool:**
- `POST /api/calculate_transformation`: Learn transformation from word pairs (e.g., `[["cow","beef"], ["pig","pork"]]`)
- `POST /api/apply_transformation`: Apply learned transformation to a new word
- `GET /api/list_examples`: List available CSV files in `vectors/` directory
- `GET /api/load_example/<filename>`: Load word pairs from CSV file

CSV files in `vectors/` directory provide pre-defined examples (animal_to_food.csv, country_to_capital.csv, etc.).

### Frontend Architecture

Two single-page applications with vanilla JavaScript:

**index.html (Word Equations):**
- Direct equation input with operator parsing
- Real-time word validation via `/api/check_word`
- Interactive 2D visualization using Plotly
- Calculation history (up to 20 entries)

**transform.html (Semantic Concepts):**
- Textarea for word pair input (CSV format)
- Pre-built example loading from `vectors/*.csv`
- Two-phase workflow: calculate transformation → apply to words
- Shows outliers removed during robust averaging

### Embeddings Files

The `download_embeddings.py` script generates three files in `embeddings/`:

| File | Size | Purpose |
|------|------|---------|
| `crawl-300d-2M-subword.bin` | ~7GB | Full FastText model for OOV words (kept for runtime use) |
| `fasttext.crawl-300d-2M.npz` | ~600MB | Pre-computed normalized vectors (500K words × 300D) |
| `fasttext.crawl-300d-2M.2d.npz` | ~8MB | 2D PCA projection + transformation parameters |

The download script:
1. Downloads from Facebook AI Research
2. Extracts top 500K words by frequency
3. Normalizes all vectors (for cosine similarity via dot product)
4. Computes PCA reduction to 2D with saved transformation parameters

## Key Implementation Details

### Performance Characteristics
- **Startup time**: 2-3 seconds for pre-computed embeddings, 30-60 seconds for FastText model
- **Similarity search**: O(n) where n=500K vocabulary size, ~50-100ms per query
- **OOV word generation**: ~5-10ms per word via FastText subwords
- **Vectorized operations**: All similarity computations use numpy matrix operations

### Windows Compatibility
The `requirements.txt` uses `fasttext-wheel` instead of `fasttext` to avoid MSVC compilation issues on Windows. The package is functionally identical.

### Error Handling
- Empty equations return "Empty equation" error
- Unknown words (when FastText unavailable) return "Word not found: '{token}'"
- OOV words (when FastText available) work normally and are reported in `oov_words` response field
- Invalid equation syntax returns "Invalid equation syntax: {error}"

### Security Notes
- `/api/load_example` validates filenames to prevent path traversal (server.py:546-547)
- `parse_and_evaluate()` uses `eval()` but restricts namespace to word vectors + numpy only (server.py:283-289)
- CORS is enabled for all origins via flask-cors

## Docker

```bash
# Build (~8GB image due to FastText model)
docker build -t vector-playground .

# Run
docker run -p 5000:5000 vector-playground

# Or use docker-compose
docker-compose up
```

The Docker image is large because it includes the full FastText model for OOV support.
