echo "Project out"
for layer in $(seq 1 12); do
python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100 \
  --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only --probe-path workspaces/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} \
  --probe-rank ${PROBE_RANK} --layer-idx $layer --no-mask-first >> logs/ent-layer-loop-out-${TASK}-r${PROBE_RANK}-l${LAYER};
done

echo "Project onto"
for layer in $(seq 1 12); do
python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100 \
  --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only --probe-path workspaces/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} \
  --probe-rank ${PROBE_RANK} --layer-idx $layer --no-mask-first --inverse >> logs/ent-layer-loop-onto-${TASK}-r${PROBE_RANK}-l${LAYER};
done