#!/usr/bin/env bash
# Independent full-model check against karpathy/llama2.c run.c.
#
#   tools/llama2c_reference/run_reference_check.sh <checkpoint> [n_tokens] [build_dir]
#
# 1. Fetches run.c at a pinned commit (needs network the first time).
# 2. Builds harness.c (which #includes run.c) with gcc.
# 3. Runs our test_model_forward_cuda with a logits dump for the same
#    deterministic token sequence (token_i = (i * 7919 + 42) % vocab).
# 4. Compares the two dumps with compare_logits.py.
#
# Expected agreement is ~1e-4 absolute, not bitwise: run.c recomputes RoPE
# angles with cosf/sinf instead of reading the stored float tables, and
# both sides accumulate in float in different orders.  The pass threshold
# (1e-2 absolute, same argmax) catches format/convention errors, which
# would show up as O(1) differences.
set -euo pipefail

CKPT=${1:?checkpoint path required}
N=${2:-8}
BUILD=${3:-build}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK="$BUILD/llama2c_reference"
COMMIT=350e04fe35433e6d2941dce5a1f53308f87058eb
mkdir -p "$WORK"

if [ ! -f "$WORK/run.c" ]; then
    curl -sSfL -o "$WORK/run.c" "https://raw.githubusercontent.com/karpathy/llama2.c/$COMMIT/run.c"
fi
sha256sum "$WORK/run.c" | tee "$WORK/run.c.sha256"

gcc -O2 -std=gnu11 -I"$WORK" -o "$WORK/harness" "$HERE/harness.c" -lm

VOCAB=32000
TOKENS=()
for ((i = 0; i < N; i++)); do TOKENS+=( $(( (i * 7919 + 42) % VOCAB )) ); done

"$WORK/harness" "$CKPT" "${TOKENS[@]}" > "$WORK/reference_logits.txt"
"$BUILD/test_model_forward_cuda" "$CKPT" "$N" "$WORK/our_logits.txt" > "$WORK/our_run.log"
tail -1 "$WORK/our_run.log"
python3 "$HERE/compare_logits.py" "$WORK/our_logits.txt" "$WORK/reference_logits.txt"
