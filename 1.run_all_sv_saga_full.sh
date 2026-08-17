#!/bin/bash
#SBATCH --job-name=sv_mixer_eur
#SBATCH --account=nn9114k
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/sv_mixer_eur_%A_%a.out
#SBATCH --array=0-105

#set -e  # Exit on error

source /cluster/projects/nn9114k/datngu/conda/etc/profile.d/conda.sh
conda activate mixer


RESULT_DIR="saga_results_107_constrain_zero_roi"
mkdir -p ${RESULT_DIR}
mkdir -p logs

LD_DIR="/cluster/projects/nn9114k/datngu/database/HC_1000G_hg38/1kg_combined_plink_eur_AC3"
SNP_FILE="$LD_DIR/SV_variants.txt"
ANNOT="${LD_DIR}/annot_mat.txt"

SUMSTAT_DIR="/cluster/projects/nn9114k/datngu/database/alkes_group_data/sumstats_107"

# Create array of all trait files
TRAITS=(${SUMSTAT_DIR}/*.sumstats.gz)

# Get the trait for this array task
TRAIT="${TRAITS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$TRAIT" ]]; then
    echo "ERROR: Trait file not found: $TRAIT"
    exit 1
fi

BASENAME=$(basename "$TRAIT" .sumstats.gz)
OUT="${RESULT_DIR}/univar_${BASENAME}.txt"
OUT_LIST="${RESULT_DIR}/univar_${BASENAME}_list.txt"


echo "Processing trait ${SLURM_ARRAY_TASK_ID}: $BASENAME"
echo "Input: $TRAIT"
echo "Output: $OUT"

python mixer_sv.py univar \
    --annot "$ANNOT" \
    --ld-mat1 "$LD_DIR" \
    --trait1 "$TRAIT" \
    --snp-file "$SNP_FILE" \
    --output "$OUT" \
    --constrain-roi-estimates-to-zero \
    --seed 42

echo "Completed: $BASENAME"
