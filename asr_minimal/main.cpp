#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    int audio_format = 0;
    std::vector<uint8_t> payload;
};

bool read_u32(std::ifstream & fin, uint32_t & out) {
    return static_cast<bool>(fin.read(reinterpret_cast<char *>(&out), sizeof(out)));
}

bool read_u16(std::ifstream & fin, uint16_t & out) {
    return static_cast<bool>(fin.read(reinterpret_cast<char *>(&out), sizeof(out)));
}

bool load_wav(const std::string & path, WavData & wav, std::string & err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        err = "failed to open wav file";
        return false;
    }

    char riff[4] = {};
    char wave[4] = {};
    uint32_t riff_size = 0;
    if (!fin.read(riff, 4) || !read_u32(fin, riff_size) || !fin.read(wave, 4)) {
        err = "failed to read RIFF header";
        return false;
    }
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;

    while (fin && !(have_fmt && have_data)) {
        char chunk_id[4] = {};
        uint32_t chunk_size = 0;
        if (!fin.read(chunk_id, 4) || !read_u32(fin, chunk_size)) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = 0;
            uint16_t channels = 0;
            uint32_t sample_rate = 0;
            uint32_t byte_rate = 0;
            uint16_t block_align = 0;
            uint16_t bits_per_sample = 0;

            if (!read_u16(fin, audio_format) || !read_u16(fin, channels) || !read_u32(fin, sample_rate) ||
                !read_u32(fin, byte_rate) || !read_u16(fin, block_align) || !read_u16(fin, bits_per_sample)) {
                err = "failed to read fmt chunk";
                return false;
            }

            const uint32_t consumed = 16;
            if (chunk_size > consumed) {
                fin.seekg(chunk_size - consumed, std::ios::cur);
            }

            wav.audio_format = audio_format;
            wav.channels = channels;
            wav.sample_rate = static_cast<int>(sample_rate);
            wav.bits_per_sample = bits_per_sample;
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            wav.payload.resize(chunk_size);
            if (chunk_size > 0 && !fin.read(reinterpret_cast<char *>(wav.payload.data()), chunk_size)) {
                err = "failed to read data chunk";
                return false;
            }
            have_data = true;
        } else {
            fin.seekg(chunk_size, std::ios::cur);
        }

        if ((chunk_size % 2) != 0) {
            fin.seekg(1, std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) {
        err = "wav missing fmt or data chunk";
        return false;
    }

    if (wav.channels < 1 || wav.channels > 2) {
        err = "only mono/stereo wav is supported in this demo";
        return false;
    }

    if (!((wav.audio_format == 1 && wav.bits_per_sample == 16) || (wav.audio_format == 3 && wav.bits_per_sample == 32))) {
        err = "only PCM16 or float32 wav is supported in this demo";
        return false;
    }

    return true;
}

bool wav_to_mono_f32(const WavData & wav, std::vector<float> & mono, std::string & err) {
    if (wav.sample_rate != WHISPER_SAMPLE_RATE) {
        err = "sample rate must be 16000 Hz for this demo";
        return false;
    }

    const int bytes_per_sample = wav.bits_per_sample / 8;
    const int frame_size = bytes_per_sample * wav.channels;
    if (frame_size <= 0 || wav.payload.size() % frame_size != 0) {
        err = "invalid wav payload size";
        return false;
    }

    const size_t n_frames = wav.payload.size() / static_cast<size_t>(frame_size);
    mono.resize(n_frames);

    if (wav.audio_format == 1 && wav.bits_per_sample == 16) {
        const int16_t * p = reinterpret_cast<const int16_t *>(wav.payload.data());
        for (size_t i = 0; i < n_frames; ++i) {
            if (wav.channels == 1) {
                mono[i] = static_cast<float>(p[i]) / 32768.0f;
            } else {
                const float l = static_cast<float>(p[2 * i]) / 32768.0f;
                const float r = static_cast<float>(p[2 * i + 1]) / 32768.0f;
                mono[i] = 0.5f * (l + r);
            }
        }
    } else {
        const float * p = reinterpret_cast<const float *>(wav.payload.data());
        for (size_t i = 0; i < n_frames; ++i) {
            if (wav.channels == 1) {
                mono[i] = p[i];
            } else {
                mono[i] = 0.5f * (p[2 * i] + p[2 * i + 1]);
            }
        }
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: asr_minimal <model.bin> <audio.wav> [lang]\n";
        std::cerr << "Example: asr_minimal ../whisper.cpp/models/ggml-base.bin ../whisper.cpp/samples/jfk.wav en\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_path = argv[2];
    const std::string lang = (argc >= 4) ? argv[3] : "auto";

    WavData wav;
    std::string err;
    if (!load_wav(wav_path, wav, err)) {
        std::cerr << "WAV read error: " << err << "\n";
        return 2;
    }

    std::vector<float> pcmf32;
    if (!wav_to_mono_f32(wav, pcmf32, err)) {
        std::cerr << "WAV convert error: " << err << "\n";
        return 3;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    cparams.flash_attn = false;

    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize whisper context from model: " << model_path << "\n";
        return 4;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.translate = false;
    wparams.language = lang.c_str();
    wparams.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);

    const int ret = whisper_full(ctx, wparams, pcmf32.data(), static_cast<int>(pcmf32.size()));
    if (ret != 0) {
        std::cerr << "whisper_full failed with code: " << ret << "\n";
        whisper_free(ctx);
        return 5;
    }

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            std::cout << text;
        }
    }
    std::cout << "\n";

    whisper_free(ctx);
    return 0;
}
