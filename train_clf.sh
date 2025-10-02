export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"

poetry run clf_train \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT
