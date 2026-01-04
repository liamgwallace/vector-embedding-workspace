FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY download_embeddings.py .
COPY templates/ templates/

# Create embeddings directory
RUN mkdir -p embeddings

# Download and process embeddings (done at build time)
RUN python download_embeddings.py

# Expose port
EXPOSE 5000

# Run the server
CMD ["python", "server.py"]
