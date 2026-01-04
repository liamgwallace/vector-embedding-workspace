# Deployment Guide

## GitHub Container Registry Setup

The Docker image is automatically built and pushed to GitHub Container Registry (ghcr.io) when you push to the `main` branch.

### First-time setup:

1. **Make the package public** (after first build):
   - Go to your GitHub repository
   - Click "Packages" on the right sidebar
   - Click on the `vector-embedding-workspace` package
   - Click "Package settings"
   - Scroll down and click "Change visibility"
   - Select "Public"

2. **Pull the image on your server**:
   ```bash
   docker pull ghcr.io/liamgwallace/vector-embedding-workspace:latest
   ```

## Deploy with Docker Compose

On your home server:

```bash
# Download docker-compose.yml
wget https://raw.githubusercontent.com/liamgwallace/vector-embedding-workspace/main/docker-compose.yml

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

The app will be available at `http://your-server-ip:5000`

## Update to latest version

```bash
docker-compose pull
docker-compose up -d
```

That's it!
