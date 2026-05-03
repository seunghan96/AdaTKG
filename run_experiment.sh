#!/bin/bash
#
# AdaTKG — main launcher.
#
# Usage:
#   bash run_experiment.sh <model> <mode> [dataset] [gpu]
#
# Arguments:
#   model   : Base | AdaTKG-EMA | AdaTKG-GRU | AdaTKG-CrossAtt
#             (case-insensitive aliases: ema/gru/attn/base also accepted)
#   mode    : train | test
#   dataset : ICEWS14 (default) | ICEWS18 | ICEWS05-15 | GDELT
#   gpu     : 0 (default) | 1 | ...
#
# By default, the launcher reads the per-(model, dataset) best
# hyperparameter configuration from best_configs/<model>.csv (the same
# values reported in Appendix C of the paper) and trains the model with
# that single configuration. To override, pass HP env vars before the
# command, e.g.:
#     ML=15 NL=2 HD=512 NC=50 bash run_experiment.sh AdaTKG-EMA train ICEWS14 0
#
# Examples:
#   bash run_experiment.sh Base            train ICEWS14
#   bash run_experiment.sh AdaTKG-EMA      train ICEWS14 1
#   bash run_experiment.sh AdaTKG-GRU      train ICEWS18 0
#   bash run_experiment.sh AdaTKG-CrossAtt train GDELT   0

set -euo pipefail

# ================================================================
# Paths (override via env if your layout differs)
# ================================================================
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_DIR}/results}"
BEST_CFG_DIR="${PROJECT_DIR}/best_configs"

# ================================================================
# Parse arguments
# ================================================================
MODEL_RAW="${1:?Usage: bash run_experiment.sh <model> <train|test> [dataset] [gpu]}"
MODE="${2:?Usage: bash run_experiment.sh <model> <train|test> [dataset] [gpu]}"
DATASET="${3:-ICEWS14}"
GPU="${4:-0}"

# Normalize model alias.
case "${MODEL_RAW}" in
    Base|base|BASE)                                       MODEL="Base" ;;
    AdaTKG-EMA|ema|EMA|adatkg-ema|AdaTKG_EMA)             MODEL="AdaTKG-EMA" ;;
    AdaTKG-GRU|gru|GRU|adatkg-gru|AdaTKG_GRU|meta)        MODEL="AdaTKG-GRU" ;;
    AdaTKG-CrossAtt|attn|crossatt|attention|AdaTKG-Attn)  MODEL="AdaTKG-CrossAtt" ;;
    *) echo "ERROR: Unknown model '${MODEL_RAW}'."
       echo "       Use one of: Base | AdaTKG-EMA | AdaTKG-GRU | AdaTKG-CrossAtt"
       exit 1 ;;
esac

# Map paper-friendly model name to (--enhancement string, train script).
case "${MODEL}" in
    Base)              ENH=""                        SCRIPT="main.py" ;;
    AdaTKG-EMA)        ENH="--enhancement ema"       SCRIPT="main_enhanced.py" ;;
    AdaTKG-GRU)        ENH="--enhancement meta"      SCRIPT="main_enhanced.py" ;;
    AdaTKG-CrossAtt)   ENH="--enhancement attention" SCRIPT="main_enhanced.py" ;;
esac

# ================================================================
# Best-config lookup (best_configs/<model>.csv).
# CSV format (header row, then one row per dataset):
#   model,dataset,max_length,num_layers,hidden_dim,num_code
# Env vars (ML, NL, HD, NC) override the CSV values.
# ================================================================
BEST_CSV="${BEST_CFG_DIR}/${MODEL}.csv"
if [[ ! -f "${BEST_CSV}" ]]; then
    echo "ERROR: best-config CSV not found: ${BEST_CSV}"
    exit 1
fi
ROW="$(awk -F, -v ds="${DATASET}" 'NR>1 && $2==ds {print; exit}' "${BEST_CSV}")"
if [[ -z "${ROW}" ]]; then
    echo "ERROR: ${BEST_CSV} has no row for dataset=${DATASET}"
    exit 1
fi
CSV_ML=$(echo "${ROW}" | cut -d, -f3)
CSV_NL=$(echo "${ROW}" | cut -d, -f4)
CSV_HD=$(echo "${ROW}" | cut -d, -f5)
CSV_NC=$(echo "${ROW}" | cut -d, -f6)

ML="${ML:-${CSV_ML}}"
NL="${NL:-${CSV_NL}}"
HD="${HD:-${CSV_HD}}"
NC="${NC:-${CSV_NC}}"

# ================================================================
echo "============================================"
echo " Model:    ${MODEL}"
echo " Dataset:  ${DATASET}"
echo " Mode:     ${MODE}"
echo " GPU:      ${GPU}"
echo " HP:       ml=${ML}  nl=${NL}  hd=${HD}  nc=${NC}"
echo "============================================"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================
# Preprocessing
# ================================================================
cd "${PROJECT_DIR}"
PKL="data/${DATASET}/${DATASET}_T_14.pkl"
NPY="data/${DATASET}/${DATASET}_Bert_Entity_Embedding.npy"
[[ -f "${PKL}" ]] || python3 data_process.py --dataset "${DATASET}" --T 14
[[ -f "${NPY}" ]] || python3 word_embedding.py --dataset "${DATASET}" \
    --bert_model_path "${BERT_MODEL_PATH:-bert-base-uncased}"

# ================================================================
# Run
# ================================================================
TAG="${MODEL//-/_}"
EXP_ID="${TAG}_${DATASET}_ml${ML}_nl${NL}_hd${HD}_nc${NC}"
EXP_DIR="${SAVE_ROOT}/${EXP_ID}"
mkdir -p "${EXP_DIR}"

CMD="python3 ${SCRIPT} \
    --dataset ${DATASET} \
    --max_length ${ML} --num_layers ${NL} \
    --hidden_dim ${HD} --num_code ${NC} \
    --word_embedding_path data \
    --result_dir ${EXP_DIR} \
    ${ENH}"

if [[ "${MODE}" == "train" ]]; then
    eval "${CMD}" 2>&1 | tee "${EXP_DIR}/train_log.txt"
    echo "[done] results in ${EXP_DIR}"
elif [[ "${MODE}" == "test" ]]; then
    LOG="${EXP_DIR}/train_log.txt"
    [[ -f "${LOG}" ]] || LOG=$(find "${EXP_DIR}" -maxdepth 2 -name "log.txt" -print -quit 2>/dev/null || echo "")
    if [[ -n "${LOG}" && -f "${LOG}" ]]; then
        grep -E "\[Test\]|\[Test Emerging\]|\[Test Unknown\]|Emerging_MRR|Unknown_MRR|Best test" "${LOG}" \
            | tail -10 > "${EXP_DIR}/test_results.txt"
        cat "${EXP_DIR}/test_results.txt"
    else
        echo "ERROR: no log under ${EXP_DIR}; run 'train' first."
    fi
fi
