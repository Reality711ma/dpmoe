#!/bin/bash

# custom config
DATA=/data1/mazc/datasets/datasets-cz
TRAINER=DPMoE
DATASET=$1
CFG=configs/DPMoE/vit_b16_ep50_batch128_lr1e-4.yaml
TG=real_world

for SEED in 1 2 3
  do
    DIR=output/DPMDA/base2new/${DATASET}/${TG}/seed${SEED}
    if [ -d "$DIR" ]; then
          echo "Oops! The results exist at ${DIR} (so skip this job)"
          continue
    fi
    python train.py \
      --root ${DATA} \
      --DA_W ${DA_W} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file ${CFG} \
      --output-dir ${DIR} \
      --source-domains product clipart art\
      --target-domains ${TG} \
    sleep 1
  done