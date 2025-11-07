FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit env var (value is in MB)
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create Streamlit config inside the image so Streamlit reads it at startup.
# This ensures server.maxUploadSize is set even if the caller doesn't pass CLI flags.
RUN mkdir -p /app/.streamlit \
 && printf "server.maxUploadSize = 1024\nserver.port = 8080\nserver.address = \"0.0.0.0\"\n" > /app/.streamlit/config.toml

# Copy application code (including any .streamlit/config.toml in the repo if present)
COPY . /app

EXPOSE 8080

# Run Streamlit; CLI flags are redundant here but left to be explicit.
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.maxUploadSize=1024"]
