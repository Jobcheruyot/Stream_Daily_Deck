FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Streamlit config directories exist and set server.maxUploadSize
# Create both /app/.streamlit (project-level) and /root/.streamlit (home) to be safe.
RUN mkdir -p /app/.streamlit /root/.streamlit \
 && printf "server.maxUploadSize = 1024\nserver.port = 8080\nserver.address = \"0.0.0.0\"\n" > /app/.streamlit/config.toml \
 && printf "server.maxUploadSize = 1024\nserver.port = 8080\nserver.address = \"0.0.0.0\"\n" > /root/.streamlit/config.toml

# Copy project files (if your repo already contains .streamlit/config.toml it'll be used instead)
COPY . /app

EXPOSE 8080

# CLI flag guarantees the value at process start (CLI > env > config precedence).
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.maxUploadSize=1024"]
