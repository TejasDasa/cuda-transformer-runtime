#ifndef MODEL_WEIGHTS_HPP
#define MODEL_WEIGHTS_HPP

struct ModelWeights{
    const float* token_embedding_table = nullptr;
    const float* rms_att_weight = nullptr;
    const float* wq = nullptr;
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;
    const float* rms_ffn_weight = nullptr;
    const float* w1 = nullptr;
    const float* w2 = nullptr;
    const float* w3 = nullptr;
    const float* rms_final_weight = nullptr;
    const float* freq_cis_real = nullptr;
    const float* freq_cis_imag = nullptr;
    const float* wcls = nullptr;
};

#endif