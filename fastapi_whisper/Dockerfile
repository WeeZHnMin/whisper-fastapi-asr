FROM python:3.12-slim-bookworm AS asr_builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY whisper.cpp /src/whisper.cpp
COPY asr_minimal /src/asr_minimal

# Copy full shared-library sets (versioned files + symlinks) to preserve SONAME resolution.
RUN cmake -S /src/asr_minimal -B /src/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    && cmake --build /src/build -j"$(nproc)" \
    && mkdir -p /opt/export \
    && cp /src/build/asr_minimal /opt/export/asr_minimal \
    && cp -a /src/build/whispercpp/src/libwhisper.so* /opt/export/ \
    && cp -a /src/build/whispercpp/ggml/src/libggml.so* /opt/export/ \
    && cp -a /src/build/whispercpp/ggml/src/libggml-base.so* /opt/export/ \
    && cp -a /src/build/whispercpp/ggml/src/libggml-cpu.so* /opt/export/

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASR_BIN_PATH=/opt/asr/asr_minimal \
    WHISPER_MODEL_PATH=/models/ggml-base.bin \
    ASR_TIMEOUT_SEC=180

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=asr_builder /opt/export/ /opt/asr/
ENV LD_LIBRARY_PATH=/opt/asr

COPY fastapi_whisper_mvp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fastapi_whisper_mvp/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
