# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vector Embedding Playground is a Flask-based web application for exploring word vector embeddings through arithmetic operations. The application demonstrates semantic relationships between words using FastText embeddings (crawl-300d-2M-subword model).

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Download FastText embeddings (first time only, ~2GB download, ~7GB extracted)
python download_embeddings.py

# Start the development server
python server.py
```

The server runs at http://localhost:5000 with Flask debug mode enabled.

## Architecture

### Backend (server.py)

The Flask server provides four main responsibilities:

1. **Embedding Management**: Loads pre-computed FastText embeddings (500,000 words × 300 dimensions) into memory at startup. The embeddings are stored as normalized numpy float32 vectors for fast cosine similarity via dot product.

2. **OOV (Out-of-Vocabulary) Support**: Loads the full FastText binary model (~7GB) to handle typos, rare words, and novel terms via character n-gram subword vectors. Words not in the pre-computed vocabulary fall back to on-the-fly vector generation.

3. **Vector Arithmetic Engine**: The `parse_and_evaluate()` function handles equation parsing with scalar multiplication and parentheses support. Examples: `king - man + woman`, `0.8 * (woman - man) + king`.

4. **Similarity Search**: `find_nearest_neighbors()` performs vectorized cosine similarity search using numpy matrix operations with argpartition for efficient top-k selection.

### API Endpoints

- `POST /api/calculate`: Accepts equation and num_results (1-50). Returns nearest neighbors with similarity scores and 2D visualization coordinates.
- `GET /api/check_word/<word>`: Validates if a word exists in vocabulary. Returns `exists` (in vocab), `oov_supported` (can be handled via FastText), and `can_use` (usable in equations).
- `GET /api/vocab_size`: Returns vocabulary size, embedding dimensions, and OOV support status.
- `GET /api/vocab_sample`: Returns ~3000 sampled words with 2D PCA coordinates for background visualization.
- `POST /api/calculate_transformation`: Calculate transformation vector from word pairs.
- `POST /api/apply_transformation`: Apply a calculated transformation to a word.
- `GET /api/list_examples`: List available example CSV files.
- `GET /api/load_example/<filename>`: Load word pairs from a CSV file.

### Frontend (templates/index.html)

Single-page application with vanilla JavaScript. Key features:

- **Equation Input**: Supports direct word input with arithmetic operators (king - man + woman), scalar multiplication (0.5 * king), and parentheses.
- **OOV Handling**: Words not in vocabulary are still usable via FastText subword vectors. The API reports which words were handled via OOV.
- **2D Visualization**: Results are projected to 2D using PCA for interactive visualization.
- **History**: Maintains up to 20 recent calculations.

### Embeddings

The application uses FastText crawl-300d-2M-subword embeddings:

| File | Size | Purpose |
|------|------|---------|
| `crawl-300d-2M-subword.bin` | ~7GB | Full model for OOV word handling |
| `fasttext.crawl-300d-2M.npz` | ~600MB | Pre-computed normalized vectors for fast lookup |
| `fasttext.crawl-300d-2M.2d.npz` | ~8MB | 2D PCA projection for visualization |

The download script (`download_embeddings.py`) fetches the model from Facebook AI Research, extracts vocabulary vectors, normalizes them, and computes PCA reduction.

## Key Implementation Details

### OOV Word Handling
When a word is not in the pre-computed 500K vocabulary:
1. The FastText binary model generates a vector using character n-gram subwords
2. The vector is normalized to match pre-computed embeddings
3. The word can be used in equations, but won't appear in similarity results (only vocab words are searched)

### Performance Considerations
- Pre-computed embeddings are loaded once at startup (~2-3 seconds)
- FastText model load takes ~30-60 seconds but enables OOV support
- Similarity search is O(n) where n = vocabulary size (500K), using vectorized numpy operations
- OOV word vector generation is ~5-10ms per word
- Each calculation performs full vocabulary scan with argpartition for top-k

### Error Handling
- Empty equations return "Empty equation" error
- Unknown words (with no FastText model) return "Word not found: '{token}'"
- OOV words (with FastText model) work and are reported in the `oov_words` response field

## Docker

```bash
# Build (downloads ~2GB, creates ~8GB image)
docker build -t vector-playground .

# Run
docker run -p 5000:5000 vector-playground
```

Or use docker-compose:
```bash
docker-compose up
```

## Development Notes

- No test suite currently exists
- Flask runs with `debug=True` and binds to `0.0.0.0:5000`
- CORS is enabled for all origins via flask-cors
- The Docker image is large (~8GB) due to the FastText model; consider using volumes for the model in production
