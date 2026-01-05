"""
Vector Embedding Playground - Flask Server
Loads FastText embeddings and provides an API for vector arithmetic.
Supports OOV (out-of-vocabulary) words through FastText subword vectors.
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

# Global storage for FastText model (for OOV support)
fasttext_model = None

# Global storage for 2D PCA coordinates
embeddings_2d = None     # numpy array of shape (vocab_size, 2)
pca_mean = None          # PCA mean for transforming new vectors
pca_components = None    # PCA components for transforming new vectors

# Global storage for transformation vector
current_transformation = None  # Stores the currently calculated transformation vector


def load_fasttext_model(filepath):
    """Load FastText binary model for OOV word support."""
    global fasttext_model
    import fasttext

    print(f"Loading FastText model from {filepath}...")
    print("  (This provides OOV/typo support via subword vectors)")
    fasttext_model = fasttext.load_model(filepath)
    print(f"  FastText model loaded successfully")


def load_embeddings(filepath):
    """Load pre-processed embeddings from .npz file."""
    global embedding_matrix, words_list, word_to_idx, embedding_dim
    print(f"Loading pre-computed embeddings from {filepath}...")

    # Load the .npz file
    data = np.load(filepath)
    embedding_matrix = data['embeddings']
    words_list = data['words']

    # Build word -> index mapping
    word_to_idx = {word: idx for idx, word in enumerate(words_list)}
    embedding_dim = embedding_matrix.shape[1]

    print(f"  Loaded {len(words_list)} words with {embedding_dim} dimensions")


def load_pca_embeddings(filepath):
    """Load pre-computed 2D PCA embeddings and transformation parameters from .npz file."""
    global embeddings_2d, pca_mean, pca_components
    print(f"Loading 2D PCA embeddings from {filepath}...")

    try:
        data = np.load(filepath)
        embeddings_2d = data['embeddings_2d']
        pca_mean = data.get('pca_mean', None)
        pca_components = data.get('pca_components', None)
        print(f"  Loaded 2D coordinates for {len(embeddings_2d)} words")
        if pca_mean is not None and pca_components is not None:
            print(f"  Loaded PCA transformation parameters")
    except FileNotFoundError:
        print(f"Warning: 2D PCA embeddings not found at {filepath}")
        print("Visualization will not be available. Run download_embeddings.py to generate them.")
        embeddings_2d = None
        pca_mean = None
        pca_components = None


def get_vector(word):
    """
    Get the vector for a word.
    First checks pre-computed vocabulary for fast lookup.
    Falls back to FastText model for OOV words (typos, rare words).
    Returns None only if FastText model is not loaded and word is OOV.
    """
    word_lower = word.lower()

    # First, try pre-computed vocabulary (fast lookup)
    idx = word_to_idx.get(word_lower)
    if idx is not None:
        return embedding_matrix[idx].copy()

    # Fall back to FastText model for OOV words
    if fasttext_model is not None:
        vector = fasttext_model.get_word_vector(word_lower).astype(np.float32)
        # Normalize the vector to match pre-computed embeddings
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    return None


def is_oov(word):
    """Check if a word is out-of-vocabulary (not in pre-computed embeddings)."""
    return word.lower() not in word_to_idx


def get_2d_coords(word):
    """Get the 2D PCA coordinates for a word, returns None if not found or not loaded."""
    if embeddings_2d is None:
        return None
    idx = word_to_idx.get(word.lower())
    if idx is None:
        return None
    return embeddings_2d[idx]


def project_vector_to_2d(vector):
    """
    Project an arbitrary vector to 2D using the saved PCA transformation.
    Returns None if PCA parameters are not loaded.
    """
    if pca_mean is None or pca_components is None:
        return None

    # Normalize the vector first (since our embeddings are normalized)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Apply PCA transformation: (vector - mean) @ components.T
    vector_centered = vector - pca_mean
    vector_2d = vector_centered @ pca_components.T

    return vector_2d


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
    Parse an equation with scalar multiplication and parentheses support.
    Examples: "king - man + woman", "0.8 * (woman - man) + king", "2 * woman - 0.5 * man"

    Returns: (result_vector, processed_equation, list_of_words_used, error_message, oov_words)
    """
    # Normalize the equation
    equation = equation.strip()
    if not equation:
        return None, "", [], "Empty equation", []

    # Replace register names with their values
    sorted_registers = sorted(registers.items(), key=lambda x: len(x[0]), reverse=True)
    processed_equation = equation
    for reg_name, reg_value in sorted_registers:
        if reg_value:
            pattern = r'\b' + re.escape(reg_name) + r'\b'
            processed_equation = re.sub(pattern, reg_value.lower(), processed_equation, flags=re.IGNORECASE)

    # Extract all words (tokens that are not numbers, operators, or parentheses)
    # Words are sequences of letters
    word_pattern = r'\b[a-zA-Z]+\b'
    words = re.findall(word_pattern, processed_equation)
    words_used = []
    oov_words = []

    # Create a safe namespace with vectors for each word
    namespace = {}
    for word in words:
        word_lower = word.lower()
        vec = get_vector(word_lower)
        if vec is None:
            return None, "", [], f"Word not found: '{word}'", []

        # Track OOV words
        if is_oov(word_lower):
            oov_words.append(word_lower)

        # Use the original word as variable name for case-insensitive matching
        namespace[word_lower] = vec.copy()
        words_used.append(word_lower)

    # Remove duplicates from words_used and oov_words
    words_used = list(dict.fromkeys(words_used))
    oov_words = list(dict.fromkeys(oov_words))

    # Replace words in equation with lowercase versions for evaluation
    eval_equation = processed_equation
    for word in words:
        # Replace each word with its lowercase version
        eval_equation = re.sub(r'\b' + re.escape(word) + r'\b', word.lower(), eval_equation, flags=re.IGNORECASE)

    # Add numpy to namespace for array operations
    namespace['__builtins__'] = {}
    import numpy as np
    namespace['np'] = np

    try:
        # Evaluate the expression safely
        result = eval(eval_equation, namespace)

        # Ensure result is a numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.float32)

        return result, eval_equation, words_used, None, oov_words

    except SyntaxError as e:
        return None, "", [], f"Invalid equation syntax: {str(e)}", []
    except Exception as e:
        return None, "", [], f"Error evaluating equation: {str(e)}", []


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
    result_vector, processed_equation, words_used, error, oov_words = parse_and_evaluate(equation, registers)

    if error:
        return jsonify({'success': False, 'error': error})

    # Find nearest neighbors
    neighbors = find_nearest_neighbors(result_vector, n=num_results, exclude_words=words_used)

    # Build results with 2D coordinates
    results = []
    for word, sim in neighbors:
        result_item = {'word': word, 'similarity': round(sim, 4)}
        coords = get_2d_coords(word)
        if coords is not None:
            result_item['x'] = float(coords[0])
            result_item['y'] = float(coords[1])
        results.append(result_item)

    # Get 2D coordinates for input words
    input_words_2d = {}
    for word in words_used:
        coords = get_2d_coords(word)
        if coords is not None:
            input_words_2d[word] = {'word': word, 'x': float(coords[0]), 'y': float(coords[1])}

    # Project the result vector to 2D
    result_vector_2d = project_vector_to_2d(result_vector)
    result_coords = None
    if result_vector_2d is not None:
        result_coords = {'x': float(result_vector_2d[0]), 'y': float(result_vector_2d[1])}

    return jsonify({
        'success': True,
        'results': results,
        'equation': processed_equation,
        'input_words': input_words_2d,
        'result_vector': result_coords,
        'oov_words': oov_words  # Report which words were OOV (handled via FastText subwords)
    })


@app.route('/api/check_word/<word>')
def check_word(word):
    """
    Check if a word exists in the vocabulary.
    Returns whether the word is in pre-computed vocab and whether it can be handled via OOV.
    """
    word_lower = word.lower()
    in_vocab = word_lower in word_to_idx
    can_handle = in_vocab or fasttext_model is not None

    return jsonify({
        'word': word,
        'exists': in_vocab,
        'oov_supported': can_handle and not in_vocab,
        'can_use': can_handle
    })


@app.route('/api/vocab_size')
def vocab_size():
    """Return the vocabulary size."""
    return jsonify({
        'size': len(words_list),
        'dimensions': embedding_dim,
        'oov_support': fasttext_model is not None
    })


@app.route('/api/vocab_sample')
def vocab_sample():
    """Return a sample of vocabulary words with 2D coordinates for background visualization."""
    if embeddings_2d is None:
        return jsonify({'success': False, 'error': '2D embeddings not available'})

    # Sample every Nth word to get approximately 2000-5000 words for background
    sample_rate = max(1, len(words_list) // 3000)

    sample_words = []
    for i in range(0, len(words_list), sample_rate):
        word = words_list[i]
        coords = embeddings_2d[i]
        sample_words.append({
            'word': word,
            'x': float(coords[0]),
            'y': float(coords[1])
        })

    return jsonify({
        'success': True,
        'words': sample_words,
        'count': len(sample_words)
    })


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
    # Look for FastText model file (for OOV support)
    model_paths = [
        'embeddings/crawl-300d-2M-subword.bin',
        'crawl-300d-2M-subword.bin',
        os.path.expanduser('~/crawl-300d-2M-subword.bin')
    ]

    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break

    if model_file:
        load_fasttext_model(model_file)
    else:
        print("\nWarning: FastText model not found. OOV word support will be disabled.")
        print("Words not in vocabulary will return errors.")

    # Look for pre-computed embeddings file
    embedding_paths = [
        'embeddings/fasttext.crawl-300d-2M.npz',
        'fasttext.crawl-300d-2M.npz',
        os.path.expanduser('~/fasttext.crawl-300d-2M.npz')
    ]

    embedding_file = None
    for path in embedding_paths:
        if os.path.exists(path):
            embedding_file = path
            break

    if embedding_file is None:
        print("=" * 60)
        print("ERROR: Processed FastText embeddings not found!")
        print("Please run: python download_embeddings.py")
        print("=" * 60)
        exit(1)

    load_embeddings(embedding_file)

    # Load 2D PCA embeddings for visualization
    pca_paths = [
        'embeddings/fasttext.crawl-300d-2M.2d.npz',
        'fasttext.crawl-300d-2M.2d.npz',
        os.path.expanduser('~/fasttext.crawl-300d-2M.2d.npz')
    ]

    pca_file = None
    for path in pca_paths:
        if os.path.exists(path):
            pca_file = path
            break

    if pca_file:
        load_pca_embeddings(pca_file)
    else:
        print("\nWarning: 2D PCA embeddings not found. Visualization will be limited.")
        print("Run 'python download_embeddings.py' to generate them if needed.")

    print("\nStarting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
