# TrainVoice ASR Service

FastAPI + whisper.cpp + custom `asr_minimal` binary speech-to-text service.

- Single-file transcription: `POST /transcribe`
- Batch transcription: `POST /transcribe/batch`
- Multi-format audio upload support (`wav/mp3/m4a/flac/ogg/aac`), normalized by `ffmpeg` to 16k mono WAV

## Project Structure

- `fastapi_whisper/`: API service, Docker config, model mount point
- `asr_minimal/`: C++ inference entry binary
- `whisper.cpp/`: upstream dependency (configured as git submodule)

## Prerequisites

- Docker Desktop (or Docker Engine + Compose)
- Model file at:
  - `fastapi_whisper/models/ggml-base.bin`

## Clone

If cloning from remote, include submodule:

```bash
git clone --recurse-submodules <your-repo-url>
```

If already cloned without submodule:

```bash
git submodule update --init --recursive
```

## Start With Docker Compose

```bash
cd fastapi_whisper
docker compose up --build
```

Service will listen on: `http://127.0.0.1:8000`

## API

### 1) Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{"status":"ok"}
```

### 2) Single File Transcription

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@C:/path/to/audio.m4a" \
  -F "language=zh"
```

Response:

```json
{
  "text": "...",
  "language": "zh",
  "filename": "audio.m4a"
}
```

### 3) Batch Transcription

Use form key `files` repeatedly:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/batch" \
  -F "files=@C:/path/to/a.mp3" \
  -F "files=@C:/path/to/b.m4a" \
  -F "language=auto"
```

Response:

```json
{
  "results": [
    {"filename": "a.mp3", "ok": true, "text": "...", "language": "auto"},
    {"filename": "b.m4a", "ok": false, "error": {"code": "audio_decode_failed", "message": "Failed to decode or transcode uploaded audio"}}
  ],
  "summary": {"total": 2, "success": 1, "failed": 1, "max_concurrency": 6}
}
```

## Key Environment Variables

Set in `fastapi_whisper/docker-compose.yml` (or runtime env):

- `WHISPER_MODEL_PATH` default: `/models/ggml-base.bin`
- `ASR_TIMEOUT_SEC` default: `180`
- `ASR_MAX_UPLOAD_MB` default: `25`
- `ASR_ALLOWED_EXTENSIONS` default: `.wav,.mp3,.m4a,.flac,.ogg,.aac`
- `ASR_ENABLE_TRANSCODE` default: `1`
- `ASR_TRANSCODE_TIMEOUT_SEC` default: `60`
- `ASR_TRANSCODE_SAMPLE_RATE` default: `16000`
- `ASR_MAX_FILES_PER_REQUEST` default: `5`
- `ASR_MAX_CONCURRENCY` default: `cpu_count/2`

## Common Error Codes

- `unsupported_audio_format` (415)
- `file_too_large` (413)
- `audio_decode_failed` (422)
- `audio_transcode_timeout` (504)
- `asr_timeout` (504)
- `too_many_files` (400)

## Stop Service

```bash
cd fastapi_whisper
docker compose down
```
