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
embedding_matrix = None  # numpy array of shape (vocab_size, embedding_dim)
words_list = None        # list of words corresponding to rows in embedding_matrix
word_to_idx = {}         # dict mapping word -> index in embedding_matrix
embedding_dim = 0

# Global storage for transformation vector
current_transformation = None  # Stores the currently calculated transformation vector


def load_glove_embeddings(filepath):
    """Load pre-processed GloVe embeddings from .npz file."""
    global embedding_matrix, words_list, word_to_idx, embedding_dim
    print(f"Loading embeddings from {filepath}...")

    # Load the .npz file
    data = np.load(filepath)
    embedding_matrix = data['embeddings']
    words_list = data['words']

    # Build word -> index mapping
    word_to_idx = {word: idx for idx, word in enumerate(words_list)}
    embedding_dim = embedding_matrix.shape[1]

    print(f"Loaded {len(words_list)} words with {embedding_dim} dimensions")


def get_vector(word):
    """Get the vector for a word, returns None if not found."""
    idx = word_to_idx.get(word.lower())
    if idx is None:
        return None
    return embedding_matrix[idx]


def find_nearest_neighbors(vector, n=10, exclude_words=None):
    """
    Find the n nearest neighbors to a vector using vectorized operations.
    Since embeddings are pre-normalized, cosine similarity = dot product.
    """
    # Normalize the query vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Compute similarities for all words at once (vectorized dot product)
    similarities = embedding_matrix @ vector  # Shape: (vocab_size,)

    # Handle excluded words
    if exclude_words:
        exclude_words = set(w.lower() for w in exclude_words)
        for word in exclude_words:
            idx = word_to_idx.get(word)
            if idx is not None:
                similarities[idx] = -np.inf  # Exclude by setting to very low value

    # Get top n indices using argpartition (faster than full sort for large arrays)
    if n < len(similarities):
        # argpartition is O(n) instead of O(n log n) for full sort
        top_indices = np.argpartition(similarities, -n)[-n:]
        # Sort just the top n
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    else:
        top_indices = np.argsort(similarities)[::-1][:n]

    # Build result list
    results = [(words_list[idx], float(similarities[idx])) for idx in top_indices]
    return results


def calculate_average_transformation(pairs):
    """
    Calculate the average transformation vector from word pairs using robust averaging.
    Returns: (avg_transform, valid_pairs, outliers_removed, outlier_pairs)
    """
    transformation_vectors = []
    valid_pairs = []

    for word_from, word_to in pairs:
        vec_from = get_vector(word_from)
        vec_to = get_vector(word_to)

        if vec_from is None or vec_to is None:
            continue

        # Calculate transformation vector (to - from)
        transform = vec_to - vec_from
        transformation_vectors.append(transform)
        valid_pairs.append((word_from, word_to))

    if not transformation_vectors:
        return None, [], 0, []

    # Convert to numpy array for easier manipulation
    transforms_array = np.array(transformation_vectors)

    # Robust averaging with outlier removal
    # Calculate pairwise similarities between all transformation vectors
    norms = np.linalg.norm(transforms_array, axis=1, keepdims=True)
    normalized_transforms = transforms_array / (norms + 1e-10)

    # Calculate similarity matrix
    similarity_matrix = normalized_transforms @ normalized_transforms.T

    # For each transform, calculate its average similarity to all others
    avg_similarities = similarity_matrix.mean(axis=1)

    # Remove outliers: keep only those with above-median average similarity
    median_similarity = np.median(avg_similarities)
    inlier_mask = avg_similarities >= median_similarity * 0.8  # 80% of median

    inlier_transforms = transforms_array[inlier_mask]
    outlier_pairs = [valid_pairs[i] for i, is_inlier in enumerate(inlier_mask) if not is_inlier]

    # Average the inliers
    avg_transform = np.mean(inlier_transforms, axis=0)

    return avg_transform, valid_pairs, len(outlier_pairs), outlier_pairs


def parse_and_evaluate(equation, registers):
    """
    Parse an equation like "A - B + C" or "king - man + woman"
    and evaluate it using vector arithmetic.

    Returns: (result_vector, processed_equation, list_of_words_used, error_message)
    """
    # Normalize the equation
    equation = equation.strip()
    if not equation:
        return None, "", [], "Empty equation"

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
        return None, "", [], "No valid tokens in equation"

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
                return None, "", [], f"Word not found: '{token}'"

            words_used.append(word)

            if result is None:
                result = vec.copy()
            elif current_op == '+':
                result = result + vec
            elif current_op == '-':
                result = result - vec

    if result is None:
        return None, "", [], "Could not evaluate equation"

    return result, processed_equation, words_used, None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/transform')
def transform():
    """Serve the transformation page."""
    return render_template('transform.html')


@app.route('/api/calculate_transformation', methods=['POST'])
def api_calculate_transformation():
    """
    Calculate transformation vector from word pairs.

    Expected JSON body:
    {
        "pairs": [["cow", "beef"], ["pig", "pork"], ...]
    }
    """
    global current_transformation

    data = request.json
    pairs = data.get('pairs', [])

    if not pairs:
        return jsonify({'success': False, 'error': 'No pairs provided'})

    # Calculate average transformation
    avg_transform, valid_pairs, outliers_removed, outlier_pairs = calculate_average_transformation(pairs)

    if avg_transform is None:
        return jsonify({'success': False, 'error': 'No valid word pairs found in vocabulary'})

    # Store the transformation
    current_transformation = avg_transform

    # Find words closest to the transformation vector itself
    transform_neighbors = find_nearest_neighbors(avg_transform, n=5)

    return jsonify({
        'success': True,
        'total_pairs': len(pairs),
        'valid_pairs': len(valid_pairs),
        'outliers_removed': outliers_removed,
        'outliers': outlier_pairs,
        'transform_words': [{'word': word, 'similarity': sim} for word, sim in transform_neighbors]
    })


@app.route('/api/apply_transformation', methods=['POST'])
def api_apply_transformation():
    """
    Apply the current transformation to a word.

    Expected JSON body:
    {
        "word": "goat"
    }
    """
    global current_transformation

    if current_transformation is None:
        return jsonify({'success': False, 'error': 'No transformation calculated yet'})

    data = request.json
    word = data.get('word', '').strip()

    if not word:
        return jsonify({'success': False, 'error': 'No word provided'})

    # Get vector for input word
    input_vec = get_vector(word)

    if input_vec is None:
        return jsonify({'success': False, 'error': f'Word "{word}" not found in vocabulary'})

    # Apply transformation
    result_vec = input_vec + current_transformation

    # Find nearest neighbors
    neighbors = find_nearest_neighbors(result_vec, n=5, exclude_words=[word])

    return jsonify({
        'success': True,
        'results': [{'word': word, 'similarity': sim} for word, sim in neighbors]
    })


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """
    Calculate the result of a vector equation.

    Expected JSON body:
    {
        "equation": "king - man + woman",
        "num_results": 10
    }
    """
    data = request.json
    equation = data.get('equation', '')
    registers = {}  # No longer using registers
    num_results = data.get('num_results', 10)

    # Validate num_results
    try:
        num_results = int(num_results)
        num_results = max(1, min(50, num_results))  # Clamp between 1 and 50
    except (ValueError, TypeError):
        num_results = 10

    # Parse and evaluate the equation
    result_vector, processed_equation, words_used, error = parse_and_evaluate(equation, registers)

    if error:
        return jsonify({'success': False, 'error': error})

    # Find nearest neighbors
    neighbors = find_nearest_neighbors(result_vector, n=num_results, exclude_words=words_used)

    return jsonify({
        'success': True,
        'results': [{'word': word, 'similarity': round(sim, 4)} for word, sim in neighbors],
        'equation': processed_equation
    })


@app.route('/api/check_word/<word>')
def check_word(word):
    """Check if a word exists in the vocabulary."""
    exists = word.lower() in word_to_idx
    return jsonify({'word': word, 'exists': exists})


@app.route('/api/vocab_size')
def vocab_size():
    """Return the vocabulary size."""
    return jsonify({'size': len(words_list), 'dimensions': embedding_dim})


@app.route('/api/list_examples')
def list_examples():
    """List available example CSV files from the vectors folder."""
    vectors_dir = 'vectors'

    if not os.path.exists(vectors_dir):
        return jsonify({'examples': []})

    examples = []
    for filename in os.listdir(vectors_dir):
        if filename.endswith('.csv'):
            # Remove .csv extension and format name
            name = filename[:-4]
            # Convert underscores to spaces and title case
            display_name = name.replace('_', ' ').title()
            examples.append({
                'filename': filename,
                'name': display_name
            })

    # Sort alphabetically by display name
    examples.sort(key=lambda x: x['name'])

    return jsonify({'examples': examples})


@app.route('/api/load_example/<filename>')
def load_example(filename):
    """Load CSV data from a specific example file."""
    vectors_dir = 'vectors'
    filepath = os.path.join(vectors_dir, filename)

    # Security check: ensure filename doesn't contain path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'success': False, 'error': 'Invalid filename'})

    if not filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'File must be a CSV'})

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    pairs.append([parts[0].strip(), parts[1].strip()])

        return jsonify({'success': True, 'pairs': pairs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Look for embeddings file
    embedding_paths = [
        'embeddings/glove.6B.50d.npz',
        'glove.6B.50d.npz',
        os.path.expanduser('~/glove.6B.50d.npz')
    ]

    embedding_file = None
    for path in embedding_paths:
        if os.path.exists(path):
            embedding_file = path
            break

    if embedding_file is None:
        print("=" * 60)
        print("ERROR: Processed GloVe embeddings not found!")
        print("Please run: python download_embeddings.py")
        print("=" * 60)
        exit(1)

    load_glove_embeddings(embedding_file)

    print("\nStarting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
