#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBES=("lensing" "clustering" "2x2pt" "combined")

MLP="weight_decay"
VMIM="gmm"
MODEL_NAME="v6"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

for PROBE in "${PROBES[@]}"; do
    OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
    DLSS_CONFIG="$REPOS/y3-deep-lss/configs/$VERSION/$SUBVERSION/$PROBE/dlss.yaml"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --environment=tensorflow \
        --output="${LOG}_training.log" \
        python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --dlss_config="$DLSS_CONFIG" \
            --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml" \
            --vmim_config="$REPOS/y3-deep-lss/configs/vmim/${VMIM}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$OUTPUT" \
            --model_name="$MODEL_NAME" \
            --include_grid \
            --include_des \
            --include_bench &
done

wait

echo "Starting Inference..."

for PROBE in "${PROBES[@]}"; do
    OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
    DLSS_CONFIG="$REPOS/y3-deep-lss/configs/$VERSION/$SUBVERSION/$PROBE/dlss.yaml"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_cls_inference.py \
            --msfm_config=\"$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml\" \
            --dlss_config=\"$DLSS_CONFIG\" \
            --out_dir=\"$OUTPUT\" \
            --model_name=\"$MODEL_NAME\" \
            --include_grid \
            --include_des \
            --include_bench" &
done

wait
