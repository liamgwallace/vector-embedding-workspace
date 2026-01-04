# Vector Embedding Playground

Explore word meanings through two interactive tools: word equations and pattern learning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and process GloVe embeddings (~862MB download, processes to ~75MB)
python download_embeddings.py

# 3. Start the server
python server.py

# 4. Open http://localhost:5000
```

## Tools

### Word Equations
Build equations with words using registers and operators:
- `king - man + woman ≈ queen`
- `paris - france + japan ≈ tokyo`
- `walked - walk + swim ≈ swam`

### Pattern Learning
Learn transformation patterns from word pair examples:
- Enter pairs like `cow,beef` and `pig,pork`
- System learns the "animal → food" pattern
- Apply to new words: `chicken → pork, fried, grilled`

Try the built-in examples: Animal→Food or Country→Capital

## How It Works

Uses [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings (400,000 words, 50 dimensions). Words are represented as vectors in semantic space. The pattern learning tool uses robust averaging with automatic outlier detection to learn transformation vectors from examples.
