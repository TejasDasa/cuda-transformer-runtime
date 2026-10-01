/*
 * Independent full-model reference for the Stories15M checkpoint.
 *
 * Provenance: this harness includes run.c from karpathy/llama2.c
 * unmodified (commit 350e04fe35433e6d2941dce5a1f53308f87058eb, fetched by
 * run_reference_check.sh).  run.c reads the same legacy checkpoint format
 * (it skips the stored freq_cis tables and recomputes RoPE angles with
 * cosf/sinf, adjacent-pair rotation, RMSNorm epsilon 1e-5).
 *
 * Adaptation: run.c's main() is renamed via the preprocessor so this file
 * can supply its own main, which calls run.c's forward() for a fixed
 * token sequence and prints raw logits as text:
 *     pos token logit0 logit1 ... logit(vocab-1)
 * No tokenizer or sampler is used.
 */
#define main llama2c_main
#include "run.c"
#undef main

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <checkpoint> <token> [token ...]\n", argv[0]);
        return 1;
    }
    Transformer t;
    build_transformer(&t, argv[1]);
    const int vocab = t.config.vocab_size;
    for (int pos = 0; pos + 2 < argc; pos++) {
        const int token = atoi(argv[pos + 2]);
        if (token < 0 || token >= vocab || pos >= t.config.seq_len) {
            fprintf(stderr, "token/position out of range\n");
            free_transformer(&t);
            return 1;
        }
        float* logits = forward(&t, token, pos);
        printf("%d %d", pos, token);
        for (int v = 0; v < vocab; v++) {
            printf(" %.9g", logits[v]);
        }
        printf("\n");
    }
    free_transformer(&t);
    return 0;
}
