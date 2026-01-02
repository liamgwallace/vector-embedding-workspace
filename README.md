# Vector Embedding Playground

A simple web tool for exploring word vector embeddings through arithmetic operations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download GloVe embeddings (~862MB download, extracts to ~170MB)
python download_embeddings.py

# 3. Start the server
python server.py

# 4. Open http://localhost:5000 in your browser
```

## Features

- **Registers**: Assign words to variables (A, B, C...) for easy reuse
- **Equation input**: Write vector equations like `A - B + C` or `king - man + woman`
- **Configurable results**: Choose how many nearest neighbors to show (1-50)
- **History**: Reload previous calculations with one click
- **Word validation**: Visual feedback shows if a word exists in the vocabulary

## How It Works

Word embeddings represent words as dense vectors in high-dimensional space. Words with similar meanings are close together. Vector arithmetic reveals interesting relationships:

- `king - man + woman ≈ queen` (gender relationship)
- `paris - france + japan ≈ tokyo` (capital city relationship)
- `walked - walk + swim ≈ swam` (tense relationship)

## Embeddings

This project uses [GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation) embeddings trained on 6 billion tokens from Wikipedia and Gigaword. The 50-dimensional version includes 400,000 words.
