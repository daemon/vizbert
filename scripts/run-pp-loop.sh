python -m vizbert.run.train_pp_probe -df ${DATA} --layer-idx ${LAYER} --model bert-base-uncased --load-weights ${MODEL_WORKSPACE} -d ${TASK} --lr 1e-2 --probe-rank ${PROBE_RANK} --objective recon --workspace workspaces/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} --num-epochs ${NUM_EPOCHS} --no-mask-first -msl ${MSL} --inverse
echo "Baseline no probe"
python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100 --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only
echo "Project onto"
python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100 --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only --probe-path workspaces/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} --probe-rank ${PROBE_RANK} --layer-idx ${LAYER} --inverse --no-mask-first
# echo "Visualize"
# python -m vizbert.run.visualize_probe -df ${DATA} -d ${TASK} --load-weights ${MODEL_WORKSPACE} --model bert-base-uncased -l ${LAYER} --workspace workspaces/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} --probe-rank ${PROBE_RANK} --output-folder visuals/pp-probe-ent-${TASK}-r${PROBE_RANK}-l${LAYER} --limit 200 --inverse
