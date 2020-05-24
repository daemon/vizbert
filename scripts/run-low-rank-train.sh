python -m vizbert.run.train_lowrank_bert -df ${DATA} --model bert-base-uncased --probe-rank 5\
  --load-weights ${MODEL_WORKSPACE} -d ${TASK} --lr 1e-2 --workspace-prefix workspaces/lowrank-inc --num-epochs ${NUM_EPOCHS}\
  -msl ${MSL};
python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100\
  --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only\
  --probe-path workspaces/lowrank-inc-kqv-${TASK}-l${LAYER}-r${PROBE_RANK} --probe-rank ${PROBE_RANK} --layer-idx ${LAYER}