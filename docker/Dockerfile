# ============================================
# Stage 1: Frontend Build
# ============================================
FROM node:24-alpine AS frontend-builder

# Install pnpm
RUN npm install -g pnpm

WORKDIR /app/frontend

COPY frontend/react-antd/package.json frontend/react-antd/pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install --frozen-lockfile || pnpm install

# Copy frontend source
COPY frontend/react-antd/ ./

# Build frontend
RUN pnpm vite build

# ============================================
# Stage 2: Backend Runtime
# ============================================
FROM python:3.11-slim-bookworm AS backend

ARG TALIB_C_VERSION="0.6.4"

# Install system dependencies and TA-Lib C library in one layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential wget curl && \
    \
    # 根据架构选择 ta-lib 安装方式
    arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
        amd64|x86_64) final_arch="amd64" ;; \
        arm64|aarch64) final_arch="arm64" ;; \
        *) final_arch="" ;; \
    esac && \
    \
    if [ -n "$final_arch" ]; then \
        echo "Detected $arch, using TA-Lib $TALIB_C_VERSION .deb" && \
        TALIB_URL="https://github.com/TA-Lib/ta-lib/releases/download/v${TALIB_C_VERSION}/ta-lib_${TALIB_C_VERSION}_${final_arch}.deb" && \
        wget -q -O /tmp/ta-lib.deb "$TALIB_URL" && \
        dpkg -i /tmp/ta-lib.deb && \
        rm -f /tmp/ta-lib.deb; \
    else \
        echo "Arch $arch not supported, building TA-Lib from source" && \
        wget -q -O /tmp/ta-lib-src.tgz "https://github.com/TA-Lib/ta-lib/releases/download/v${TALIB_C_VERSION}/ta-lib-${TALIB_C_VERSION}-src.tar.gz" && \
        mkdir /tmp/ta-lib && \
        tar xf /tmp/ta-lib-src.tgz -C /tmp/ta-lib --strip-components=1 && \
        cd /tmp/ta-lib && \
        ./configure --prefix=/usr && \
        make -j$(nproc) && \
        make install && \
        libtool --finish /usr/lib && \
        rm -rf /tmp/ta-lib /tmp/ta-lib-src.tgz; \
    fi && \
    \
    # Install TA-Lib Python binding while build tools are available
    pip install --no-cache-dir TA-Lib && \
    \
    # Clean up build dependencies (keep runtime libs)
    apt-get autoremove -y build-essential wget curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./ 

# Install Python dependencies
RUN uv pip install --system --no-cache .

# Copy backend code
COPY backend/ ./backend/
COPY config/ ./config/
COPY data/ ./data/
COPY start_api.py ./

# Copy frontend build artifacts (for FastAPI static file serving)
COPY --from=frontend-builder /app/frontend/dist ./frontend/react-antd/dist

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
