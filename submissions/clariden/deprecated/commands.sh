MODEL="debug/v1"

python ../../deep_lss/apps/run_training.py \
    --dir_base=$OUTPUT \
    --dir_model=$MODEL \
    --loss_function=$LOSS \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
    --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --dlss_config="$REPOS/y3-deep-lss/configs/$VERSION/$SUBVERSION/$PROBE/dlss.yaml" \
    --net_config="$REPOS/y3-deep-lss/configs/$VERSION/$SUBVERSION/$PROBE/deepsphere.yaml" \
    --dist_strategy="$STRATEGY" \
    --mixed_precision \
    --mixed_precision_dtype="bfloat16"