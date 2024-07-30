# training script: an example
DATASET=$1
ROOT=$2
OUTPUT_DIR=$3


if [ ${DATASET} == "pneumonia" ];
then
python train.py \
 --config-file configs/LASP/vit_b16_c4_batch32_pneumonia.yaml \
 --dataset-config-file configs/datasets/pneumonia.yaml \
 --trainer XCoOp \
 --root ${ROOT} \
 --output-dir ${OUTPUT_DIR} \
 --seed 3407 \
 --resume false
fi

