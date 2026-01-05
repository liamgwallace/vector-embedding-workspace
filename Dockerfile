FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for fasttext
# These are needed to compile the fasttext library
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY download_embeddings.py .
COPY templates/ templates/
COPY vectors/ vectors/

# Create embeddings directory
RUN mkdir -p embeddings

# Download and process FastText embeddings (done at build time)
# This downloads ~2GB and creates the model files
# Note: The .bin file is ~7GB, so the final image will be large
RUN python download_embeddings.py

# Expose port
EXPOSE 5000

# Run the server
CMD ["python", "server.py"]
