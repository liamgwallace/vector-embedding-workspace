"""
Vector Embedding Playground - Flask Server
Loads GloVe embeddings and provides an API for vector arithmetic.
"""

import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global storage for embeddings
embeddings = {}
embedding_dim = 0


def load_glove_embeddings(filepath):
    """Load GloVe embeddings from file."""
    global embeddings, embedding_dim
    print(f"Loading embeddings from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vector
            if embedding_dim == 0:
                embedding_dim = len(vector)

    print(f"Loaded {len(embeddings)} words with {embedding_dim} dimensions")


def get_vector(word):
    """Get the vector for a word, returns None if not found."""
    return embeddings.get(word.lower())


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def find_nearest_neighbors(vector, n=10, exclude_words=None):
    """Find the n nearest neighbors to a vector."""
    if exclude_words is None:
        exclude_words = set()
    else:
        exclude_words = set(w.lower() for w in exclude_words)

    similarities = []
    for word, emb in embeddings.items():
        if word not in exclude_words:
            sim = cosine_similarity(vector, emb)
            similarities.append((word, float(sim)))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


def parse_and_evaluate(equation, registers):
    """
    Parse an equation like "A - B + C" or "king - man + woman"
    and evaluate it using vector arithmetic.

    Returns: (result_vector, list_of_words_used, error_message)
    """
    # Normalize the equation
    equation = equation.strip()
    if not equation:
        return None, [], "Empty equation"

    # Replace register names with their values
    # Sort by length descending to avoid partial replacements (e.g., "AB" before "A")
    sorted_registers = sorted(registers.items(), key=lambda x: len(x[0]), reverse=True)
    processed_equation = equation
    for reg_name, reg_value in sorted_registers:
        if reg_value:  # Only replace if register has a value
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(reg_name) + r'\b'
            processed_equation = re.sub(pattern, reg_value.lower(), processed_equation, flags=re.IGNORECASE)

    # Tokenize: split by operators while keeping them
    tokens = re.split(r'(\s*[\+\-]\s*)', processed_equation)
    tokens = [t.strip() for t in tokens if t.strip()]

    if not tokens:
        return None, [], "No valid tokens in equation"

    # Parse and evaluate
    result = None
    current_op = '+'
    words_used = []

    for token in tokens:
        if token in ['+', '-']:
            current_op = token
        else:
            word = token.lower()
            vec = get_vector(word)
            if vec is None:
                return None, [], f"Word not found: '{token}'"

            words_used.append(word)

            if result is None:
                result = vec.copy()
            elif current_op == '+':
                result = result + vec
            elif current_op == '-':
                result = result - vec

    if result is None:
        return None, [], "Could not evaluate equation"

    return result, words_used, None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """
    Calculate the result of a vector equation.

    Expected JSON body:
    {
        "equation": "A - B + C",
        "registers": {"A": "king", "B": "man", "C": "woman"},
        "num_results": 10
    }
    """
    data = request.json
    equation = data.get('equation', '')
    registers = data.get('registers', {})
    num_results = data.get('num_results', 10)

    # Validate num_results
    try:
        num_results = int(num_results)
        num_results = max(1, min(50, num_results))  # Clamp between 1 and 50
    except (ValueError, TypeError):
        num_results = 10

    # Parse and evaluate the equation
    result_vector, words_used, error = parse_and_evaluate(equation, registers)

    if error:
        return jsonify({'success': False, 'error': error})

    # Find nearest neighbors
    neighbors = find_nearest_neighbors(result_vector, n=num_results, exclude_words=words_used)

    return jsonify({
        'success': True,
        'results': [{'word': word, 'similarity': round(sim, 4)} for word, sim in neighbors],
        'words_used': words_used
    })


@app.route('/api/check_word/<word>')
def check_word(word):
    """Check if a word exists in the vocabulary."""
    exists = word.lower() in embeddings
    return jsonify({'word': word, 'exists': exists})


@app.route('/api/vocab_size')
def vocab_size():
    """Return the vocabulary size."""
    return jsonify({'size': len(embeddings), 'dimensions': embedding_dim})


if __name__ == '__main__':
    # Look for embeddings file
    embedding_paths = [
        'embeddings/glove.6B.50d.txt',
        'glove.6B.50d.txt',
        os.path.expanduser('~/glove.6B.50d.txt')
    ]

    embedding_file = None
    for path in embedding_paths:
        if os.path.exists(path):
            embedding_file = path
            break

    if embedding_file is None:
        print("=" * 60)
        print("ERROR: GloVe embeddings not found!")
        print("Please run: python download_embeddings.py")
        print("=" * 60)
        exit(1)

    load_glove_embeddings(embedding_file)

    print("\nStarting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
