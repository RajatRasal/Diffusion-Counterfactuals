export IMAGE_SIZE=64
export EXPN="ddim_cfg"

poetry run diffae_train \
    --size=$IMAGE_SIZE \
    --expn=$EXPN
