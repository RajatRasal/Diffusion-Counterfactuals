export DATA_NAME="celebahq"
export IMAGE_SIZE=64
export EXPN="hoge"

poetry run diffae_train \
    --data_name=$DATA_NAME \
    --size=$IMAGE_SIZE \
    --expn=$EXPN
