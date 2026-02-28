# FastAPI + whisper.cpp MVP

This is a minimal service that:
- accepts an uploaded audio file
- normalizes audio using `ffmpeg` (16kHz mono WAV)
- calls a custom C++ binary (`asr_minimal`) via subprocess
- returns plain transcription text as JSON

## 1) Build Docker image

Run from project root (`TrainVoice`):

```powershell
docker build -f fastapi_whisper_mvp/Dockerfile -t whisper-fastapi-mvp .
```

## 2) Prepare model

Put your model on host, for example:

`C:\models\ggml-base.bin`

Then mount it into container at `/models/ggml-base.bin`.

## 3) Run container

```powershell
docker run --rm -p 8000:8000 `
  -v C:\models\ggml-base.bin:/models/ggml-base.bin `
  whisper-fastapi-mvp
```

## 4) Call API

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Transcribe:

```powershell
curl -X POST "http://127.0.0.1:8000/transcribe" `
  -F "file=@whisper.cpp/samples/jfk.wav" `
  -F "language=en"
```

Batch transcribe (multiple files, concurrent processing with server-side limit):

```powershell
curl -X POST "http://127.0.0.1:8000/transcribe/batch" `
  -F "files=@audio1.m4a" `
  -F "files=@audio2.mp3" `
  -F "language=zh"
```

For Chinese:

```powershell
curl -X POST "http://127.0.0.1:8000/transcribe" `
  -F "file=@your_audio.wav" `
  -F "language=zh"
```

## Config (env)

- `ASR_MAX_UPLOAD_MB` (default: `25`)
- `ASR_ALLOWED_EXTENSIONS` (default: `.wav,.mp3,.m4a,.flac,.ogg,.aac`)
- `ASR_ENABLE_TRANSCODE` (default: `1`)
- `ASR_TRANSCODE_TIMEOUT_SEC` (default: `60`)
- `ASR_TRANSCODE_SAMPLE_RATE` (default: `16000`)
- `ASR_MAX_FILES_PER_REQUEST` (default: `5`)
- `ASR_MAX_CONCURRENCY` (default: `cpu_count/2`, minimum `1`)

## Notes

- This is an MVP: no queue, no streaming, no auth.
- It does not call `whisper-cli`; it calls your own C++ binary (`asr_minimal`) that links to `whisper` library.
- `asr_minimal` still expects WAV 16kHz input; the API now uses `ffmpeg` to convert common audio formats before inference.
- New endpoint `/transcribe/batch` processes multiple uploaded files in one request.
- For production, add:
  - request size/time limits
  - task queue
  - structured logging and metrics
  - model warmup and concurrency control
