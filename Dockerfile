# Dockerfile (in the root of the repo)

FROM python:3.11-slim

# Set workdir inside the container
WORKDIR /app

# Install system deps if needed (optional, here it's minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code and requirements
COPY backend/ ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Environment variable for Hugging Face API key
# The actual value will be passed at docker run time.
ENV HF_API_KEY=""

# Default command: run FastAPI with uvicorn
# Note: "backend.app.main:app" matches your folder structure.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
