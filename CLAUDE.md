# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vector Embedding Playground is a Flask-based web application for exploring word vector embeddings through arithmetic operations. The application demonstrates semantic relationships between words using GloVe (Global Vectors for Word Representation) embeddings.

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings (first time only, ~862MB download)
python download_embeddings.py

# Start the development server
python server.py
```

The server runs at http://localhost:5000 with Flask debug mode enabled.

## Architecture

### Backend (server.py)

The Flask server provides three main responsibilities:

1. **Embedding Management**: Loads GloVe embeddings (400,000 words × 50 dimensions) into memory at startup via `load_glove_embeddings()`. The embeddings dictionary maps lowercase words to numpy float32 vectors.

2. **Vector Arithmetic Engine**: The `parse_and_evaluate()` function handles equation parsing with register substitution. It tokenizes equations by splitting on +/- operators while preserving them, then sequentially applies vector addition/subtraction. Register names are replaced using regex word boundaries to prevent partial matches.

3. **Similarity Search**: `find_nearest_neighbors()` performs brute-force cosine similarity search across the entire vocabulary, excluding input words from results.

### API Endpoints

- `POST /api/calculate`: Accepts equation, registers dict, and num_results (1-50). Returns nearest neighbors with similarity scores.
- `GET /api/check_word/<word>`: Validates if a word exists in vocabulary.
- `GET /api/vocab_size`: Returns vocabulary size and embedding dimensions.

### Frontend (templates/index.html)

Single-page application with vanilla JavaScript. Key features:

- **Registers**: Dynamic register count (3/5/8/10 options). Each register input validates words against the vocabulary API in real-time, applying `.valid` or `.invalid` CSS classes.
- **Equation Input**: Supports both register names (A, B, C) and direct word input (king, man, woman).
- **History**: Maintains up to 20 recent calculations. Loading from history restores equation, registers, and re-runs calculation.

### Embeddings

The application searches for `embeddings/glove.6B.50d.txt` at startup. If not found, server.py exits with instructions to run download_embeddings.py. The download script fetches the full GloVe 6B zip (~862MB) from Stanford NLP, extracts only the 50d variant (~170MB), and removes the zip file.

## Key Implementation Details

### Register Replacement
Registers are sorted by length descending before replacement to handle multi-character register names correctly (e.g., "AB" before "A"). Word boundary regex prevents "KING" from matching register "K" when "K" contains "king".

### Error Handling
- Empty equations return "Empty equation" error
- Unknown words return "Word not found: '{token}'" with the exact token
- Server unavailable shows "Failed to connect to server"

### Performance Considerations
- Embeddings are loaded once at startup and kept in memory
- Similarity search is O(n) where n = vocabulary size (400K)
- No caching or indexing of similarity searches
- Each calculation performs full vocabulary scan

## Development Notes

- No test suite currently exists
- No static file directory configured (all CSS/JS inline in index.html)
- Flask runs with `debug=True` and binds to `0.0.0.0:5000`
- CORS is enabled for all origins via flask-cors
