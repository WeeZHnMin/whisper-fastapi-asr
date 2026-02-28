import os
import subprocess
from pathlib import Path


class AsrError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AsrConfigError(AsrError):
    pass


class AsrInputError(AsrError):
    pass


class AsrTimeoutError(AsrError):
    pass


class AsrRuntimeError(AsrError):
    pass


def transcribe_with_asr_binary(
    input_audio: Path,
    language: str = "auto",
    timeout_sec: int | None = None,
) -> str:
    asr_bin = Path(os.getenv("ASR_BIN_PATH", "/opt/asr/asr_minimal"))
    model_path = Path(os.getenv("WHISPER_MODEL_PATH", "/models/ggml-base.bin"))
    try:
        timeout = timeout_sec or int(os.getenv("ASR_TIMEOUT_SEC", "180"))
    except ValueError as e:
        raise AsrConfigError("asr_timeout_invalid", "ASR timeout configuration is invalid") from e

    if not asr_bin.exists():
        raise AsrConfigError("asr_bin_not_found", "ASR binary is not configured correctly")
    if not model_path.exists():
        raise AsrConfigError("asr_model_not_found", "ASR model file is not available")
    if not input_audio.exists():
        raise AsrInputError("input_audio_not_found", "Uploaded audio file is not available")

    lang = language if language else "auto"
    cmd = [str(asr_bin), str(model_path), str(input_audio), lang]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AsrTimeoutError("asr_timeout", f"ASR timed out after {timeout} seconds") from e
    except OSError as e:
        raise AsrRuntimeError("asr_exec_failed", "ASR process could not be started") from e

    if proc.returncode != 0:
        raise AsrRuntimeError(
            "asr_failed",
            f"ASR failed with exit code {proc.returncode}",
        )

    text = proc.stdout.strip()
    if not text:
        raise AsrRuntimeError("asr_empty_output", "ASR produced empty transcription output")

    return text
