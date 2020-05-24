rm -rf logs/bbu-exp-all-pp*
for i in $(seq 1 12); do
  python -m vizbert.run.train_pp_probe -df ${DATA} --layer-idx $i --model bert-base-uncased\
    --load-weights ${MODEL_WORKSPACE} -d ${TASK} --lr 1e-2 --objective recon\
    --workspace-prefix workspaces/pp-probe-recon-${TASK}-l$i --num-epochs ${NUM_EPOCHS}\
    --no-mask-first -msl ${MSL} --inverse;
#  python -m vizbert.run.finetune_classification -df ${DATA} -d ${TASK} -w ${MODEL_WORKSPACE} --num-warmup-steps 100\
#    --num-epochs 30 --lr 5e-4 -msl ${MSL} -bsz 16 --load-weights --eval-only\
#    --probe-path workspaces/pp-probe-recon-${TASK}-l$i\
#    --layer-idx $i --inverse --no-mask-first >> logs/bbu-exp-all-pp-l$i;
done
