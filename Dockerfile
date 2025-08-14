FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model for semantic splitting
RUN python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw/pdf data/raw/docx data/processed_chunks/processed_pdf data/processed_chunks/processed_docx results configs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run.py"]
