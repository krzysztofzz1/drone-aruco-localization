FROM python:3.11-slim

# Informacje o architekturze (dostępne przy buildx)
ARG TARGETPLATFORM
ARG TARGETARCH

# Debug info (opcjonalne)
RUN echo "Building for $TARGETPLATFORM ($TARGETARCH)"

# Instalacja zależności systemowych dla OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-tk \
    tk \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie plików
COPY requirements.txt .

# Instalacja bibliotek Python
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie reszty projektu
COPY src/ src/
COPY config/ config/

# Domyślna komenda
CMD ["python", "src/main.py"]