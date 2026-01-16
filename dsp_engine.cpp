#include <cmath>
#include <algorithm>

// Windows DLL Export 매크로 (stdcall 명시)
#define EXPORT extern "C" __declspec(dllexport)

static float phase = 0.0f;
static float sample_rate = 44100.0f;

EXPORT void init_engine(float sr) {
    sample_rate = sr;
    phase = 0.0f;
}

// 1. Ring Modulator
EXPORT void apply_ring_mod(float* input, float* output, int frames, float freq, float amount) {
    if (!input || !output) return;

    float phase_inc = 2.0f * 3.14159265f * freq / sample_rate;
    
    for (int i = 0; i < frames; ++i) {
        float in_sample = input[i];
        
        // 캐리어 파형 생성 (Sine Wave)
        float carrier = std::sin(phase);
        
        // 변조: (원본 * (1-amount)) + (변조음 * amount)
        // amount가 0이면 원본 100%, 1이면 변조음 100%
        // Ring Mod 공식: Input * Carrier
        float ring_signal = in_sample * carrier;
        
        output[i] = in_sample * (1.0f - amount) + ring_signal * amount;
        
        // 위상 업데이트
        phase += phase_inc;
        if (phase > 2.0f * 3.14159265f) {
            phase -= 2.0f * 3.14159265f;
        }
    }
}

// 2. Bit Crusher
EXPORT void apply_bit_crush(float* input, float* output, int frames, int bits) {
    if (!input || !output) return;

    // 비트가 너무 낮으면 소리가 아예 안 날 수 있으므로 최소값 보장
    if (bits < 1) bits = 1;
    if (bits > 32) bits = 32;

    float levels = std::pow(2.0f, (float)bits);
    
    for (int i = 0; i < frames; ++i) {
        float sample = input[i];
        // 양자화 (Quantization)
        output[i] = std::floor(sample * levels) / levels;
    }
}