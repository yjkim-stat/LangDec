export HF_TOKEN=TODO
export CACHE_DIR="/content"
export ATTN_IMPLEMENTATION='sdpa' # 'flash_attention_2'
export GDRIVE_DIR="/content/drive/MyDrive/colab_files"

# Define default arguments
MAX_NEW_TOKENS=1000

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# PRM config
PRM_MODEL_NAME="UW-Madison-Lee-Lab/VersaPRM"
POSITIVE_TAG="+"
NEGATIVE_TAG="-"

BATCH_SIZE=1
SECONDARY_DEVICE="cuda"

# Search config
SCORE_AGGREGATION="min"

MAX_TRIALS=10

# --- 최종 method 이름 생성 ---
METHOD_NAME="N${MAX_TRIALS}"

CUDA_DEVICE_ID=0

seed=1234

dataset='HuggingFaceH4/MATH-500'

start=0
end=98

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python LangDec/main.py \
    --seed $seed\
    --method "SC_${METHOD_NAME}"\
    --version V1\
    --dataset $dataset\
    --hf_token $HF_TOKEN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --secondary_device $SECONDARY_DEVICE \
    --max_trials $MAX_TRIALS\
    --score_aggregation $SCORE_AGGREGATION \
    --prm_model_name $PRM_MODEL_NAME\
    --positive_tag $POSITIVE_TAG \
    --negative_tag $NEGATIVE_TAG \
    --test_sample_idx $(seq $start $end)

