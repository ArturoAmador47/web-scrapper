# Dockerfile para Web Tech Scraper
FROM python:3.11-slim-bookworm

# Instalar dependencias del sistema para WeasyPrint y Chromium (crawl4ai)
# Compatible con ARM64 (Apple Silicon) y AMD64
RUN apt-get update && apt-get install -y --no-install-recommends \
    # WeasyPrint dependencies
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    libcairo2 \
    libgirepository1.0-dev \
    gir1.2-pango-1.0 \
    # Para crawl4ai (headless browser) - opcional, comentar si falla
    chromium \
    # Build tools
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Variables de entorno para Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMIUM_FLAGS="--no-sandbox --headless --disable-gpu --disable-dev-shm-usage"

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar playwright para crawl4ai
RUN pip install playwright && \
    playwright install chromium --with-deps || true

# Copiar el código fuente
COPY . .

# Crear directorio de output
RUN mkdir -p /app/output

# Exponer el puerto de la API
EXPOSE 8000

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Comando por defecto para ejecutar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
