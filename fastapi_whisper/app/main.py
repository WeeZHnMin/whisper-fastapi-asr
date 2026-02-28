import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .runner import (
    AsrConfigError,
    AsrError,
    AsrInputError,
    AsrRuntimeError,
    AsrTimeoutError,
    transcribe_with_asr_binary,
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


app = FastAPI(title="whisper.cpp FastAPI MVP", version="0.1.0")
CHUNK_SIZE_BYTES = 1024 * 1024
MAX_UPLOAD_MB = float(os.getenv("ASR_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ENABLE_TRANSCODE = os.getenv("ASR_ENABLE_TRANSCODE", "1").lower() in {"1", "true", "yes", "on"}
TRANSCODE_TIMEOUT_SEC = _int_env("ASR_TRANSCODE_TIMEOUT_SEC", 60)
TRANSCODE_SAMPLE_RATE = _int_env("ASR_TRANSCODE_SAMPLE_RATE", 16000)
MAX_FILES_PER_REQUEST = _int_env("ASR_MAX_FILES_PER_REQUEST", 5)
DEFAULT_MAX_CONCURRENCY = max(1, (os.cpu_count() or 1) // 2)
MAX_CONCURRENCY = _int_env("ASR_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
ASR_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
ALLOWED_EXTENSIONS = {
    (clean_ext if clean_ext.startswith(".") else f".{clean_ext}").lower()
    for ext in os.getenv("ASR_ALLOWED_EXTENSIONS", ".wav,.mp3,.m4a,.flac,.ogg,.aac").split(",")
    for clean_ext in [ext.strip()]
    if clean_ext
}


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _validate_extension(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise _http_error(
            status_code=415,
            code="unsupported_audio_format",
            message=f"Unsupported audio format. Allowed extensions: {allowed}",
        )
    return suffix


def _transcode_to_wav(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(TRANSCODE_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TRANSCODE_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError as e:
        raise _http_error(
            status_code=500,
            code="audio_transcoder_not_available",
            message="ffmpeg is not available in runtime",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise _http_error(
            status_code=504,
            code="audio_transcode_timeout",
            message=f"Audio transcoding timed out after {TRANSCODE_TIMEOUT_SEC} seconds",
        ) from e

    if proc.returncode != 0 or not output_path.exists():
        raise _http_error(
            status_code=422,
            code="audio_decode_failed",
            message="Failed to decode or transcode uploaded audio",
        )


async def _save_upload_file(file: UploadFile, destination: Path) -> int:
    written = 0
    with destination.open("wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise _http_error(
                    status_code=413,
                    code="file_too_large",
                    message=f"File exceeds maximum allowed size of {MAX_UPLOAD_MB:g} MB",
                )
            f.write(chunk)
    return written


def _run_asr_pipeline(audio_path: Path, normalized_path: Path, language: str) -> str:
    asr_input_path = audio_path
    if ENABLE_TRANSCODE:
        _transcode_to_wav(audio_path, normalized_path)
        asr_input_path = normalized_path
    return transcribe_with_asr_binary(asr_input_path, language=language)


def _translate_asr_exception(e: Exception) -> HTTPException:
    if isinstance(e, AsrTimeoutError):
        return _http_error(status_code=504, code=e.code, message=e.message)
    if isinstance(e, AsrInputError):
        return _http_error(status_code=400, code=e.code, message=e.message)
    if isinstance(e, AsrConfigError):
        return _http_error(status_code=500, code=e.code, message=e.message)
    if isinstance(e, AsrRuntimeError):
        return _http_error(status_code=502, code=e.code, message=e.message)
    if isinstance(e, AsrError):
        return _http_error(status_code=400, code=e.code, message=e.message)
    return _http_error(
        status_code=500,
        code="internal_error",
        message="Internal server error",
    )


async def _process_file(file: UploadFile, language: str) -> dict[str, str]:
    suffix = _validate_extension(file.filename)

    try:
        with tempfile.TemporaryDirectory(prefix="asr_in_") as in_dir:
            audio_path = Path(in_dir) / f"input{suffix}"
            normalized_path = Path(in_dir) / "normalized.wav"
            await _save_upload_file(file, audio_path)

            async with ASR_SEMAPHORE:
                text = await asyncio.to_thread(
                    _run_asr_pipeline,
                    audio_path,
                    normalized_path,
                    language,
                )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise _translate_asr_exception(e) from e
    finally:
        await file.close()

    return {
        "text": text,
        "language": language,
        "filename": file.filename or "",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
) -> JSONResponse:
    result = await _process_file(file, language)
    return JSONResponse(result)


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: str = Form("auto"),
) -> JSONResponse:
    if not files:
        raise _http_error(status_code=400, code="no_files", message="No files uploaded")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise _http_error(
            status_code=400,
            code="too_many_files",
            message=f"Maximum {MAX_FILES_PER_REQUEST} files per request",
        )

    tasks = [_process_file(file, language) for file in files]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[dict[str, object]] = []
    success_count = 0
    for idx, item in enumerate(raw_results):
        filename = files[idx].filename or ""
        if isinstance(item, Exception):
            if isinstance(item, HTTPException):
                detail = item.detail if isinstance(item.detail, dict) else {"message": str(item.detail)}
                error_code = str(detail.get("code", "request_failed"))
                error_message = str(detail.get("message", "Request failed"))
            else:
                error_code = "internal_error"
                error_message = "Internal server error"
            results.append(
                {
                    "filename": filename,
                    "ok": False,
                    "error": {"code": error_code, "message": error_message},
                }
            )
            continue

        success_count += 1
        results.append({"filename": filename, "ok": True, "text": item["text"], "language": item["language"]})

    return JSONResponse(
        {
            "results": results,
            "summary": {
                "total": len(files),
                "success": success_count,
                "failed": len(files) - success_count,
                "max_concurrency": MAX_CONCURRENCY,
            },
        }
    )
