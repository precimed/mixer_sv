# MiXeR-SV: Structural Variant Enrichment Analysis

> **Version 0.1** — SV enrichment analysis.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [LD Reference Data](#3-ld-reference-data)
4. [Summary Statistics](#4-summary-statistics)
5. [Usage](#5-usage)
6. [CLI Arguments](#6-cli-arguments)
7. [Output Files](#7-output-files)
8. [Citation](#8-citation)
9. [License](#9-license)
10. [Changelog](#10-changelog)

---

## 1. Overview

**MiXeR-SV** is a statistical genetics tool for quantifying the enrichment of structural variants (SVs) in trait heritability. It builds on the MiXeR framework. Key features include:

- Univariate SV enrichment analysis via LD-score regression and a Gaussian mixture heritability model.
- Annotation-based enrichment estimation with fold-enrichment statistics and standard errors.
- QC of GWAS summary statistics (MAF filtering, MHC exclusion, ambiguous SNP removal, allele alignment).
- Chromosome-level data processing for memory efficiency.

---

## 2. Installation

The recommended way to set up the environment is via [conda](https://docs.conda.io/en/latest/):

```sh
# Create a new conda environment with Python 3.13
conda create -n mixer_sv python=3.13 -y

# Activate the environment
conda activate mixer_sv

# Install all required packages
pip install -r requirements.txt
```

**Key dependencies** (see [requirements.txt](requirements.txt) for pinned versions):

| Package | Purpose |
|---|---|
| `torch` | PyTorch optimization backend |
| `numpy` / `scipy` | Numerical computing |
| `pandas` | Data handling |
| `scikit-learn` | Utility functions |
| `mat73` | Loading MATLAB v7.3 LD matrix files |


---

## 3. LD Reference Data

MiXeR-SV requires pre-computed LD matrices. We provide LD reference data derived from the **1000 Genomes Project (GRCh38)** using both WGS 30X and ONT long-read sequencing:

**Download link:**

```
https://doi.org/10.5281/zenodo.20676136
```

You can download the two tar.gz files for the EAS and EUR populations, then extract them:
```sh
## EAS
tar -xzf 1kg_combined_plink_eas_AC3.tar.gz
## EUR
tar -xzf 1kg_combined_plink_eur_AC3.tar.gz
```

The directory structure should look like:

```
1kg_combined_plink_eur_AC3/
├── SV_variants.txt          # List of SV variant IDs
├── annot_mat.txt            # Annotation matrix (tab-separated, SNP-indexed)
├── chr1.ldmat/
├── chr2.ldmat/
│   ...
└── chr22.ldmat/
```

---

## 4. Summary Statistics

MiXeR-SV reads GWAS summary statistics in the standard format used by [LDSC](https://github.com/bulik/ldsc). A curated collection of 107 independent GWAS summary statistics is available from:

```
https://zenodo.org/records/10515792/files/sumstats_indep107.tgz?download=1
```

A few example files are included in the `sumstats/` directory for quick testing.

---

## 5. Usage

### Computational requirements

| Hardware | Estimated runtime per trait |
|---|---|
| Apple M4 Max (36 GB RAM) | < 10 minutes |
| Intel CPU (32 GB+ RAM) | ~30 mins – 1 hours |

At least **32 GB of RAM** is recommended for genome-wide analysis.

### Running analysis

```sh
# Activate the conda environment
conda activate mixer_sv

# Create output directory
mkdir -p test_results

# Set paths to LD reference data (replace with your actual path)
LD_DIR="1kg_combined_plink_eur_AC3"

# other annotation files, should be in the same directory as the provided LD matrices
SNP_FILE="$LD_DIR/SV_variants.txt"
ANNOT="$LD_DIR/annot_mat.txt"

# Run analysis for all traits in sumstats/
for TRAIT in sumstats/*.sumstats.gz; do
    BASENAME=$(basename "$TRAIT" .sumstats.gz)
    OUT="test_results/univar_${BASENAME}.txt"
    OUT_LIST="test_results/univar_${BASENAME}_list.txt"

    # Skip if output already exists
    if [[ -f "$OUT_LIST" && -s "$OUT_LIST" ]]; then
        echo "Skipping (already exists): $OUT_LIST"
        continue
    fi

    echo "Processing: $BASENAME"

    python mixer_sv.py univar \
        --annot    "$ANNOT"    \
        --ld-mat1  "$LD_DIR"   \
        --trait1   "$TRAIT"    \
        --snp-file "$SNP_FILE" \
        --output   "$OUT"      \
        --seed 42
done
```

### Running the constrained model (null SV enrichment)

The **constrained model** fixes the within-region heritability coefficient to zero, effectively testing the null hypothesis that SVs contribute no additional heritability beyond the genome-wide baseline. This is useful for:


Add `--constrain-roi-estimates-to-zero` to any `univar` call:

```sh
# Activate the conda environment
conda activate mixer_sv

# Create output directory
mkdir -p test_results_constrained

# Set paths to LD reference data
LD_DIR="1kg_combined_plink_eur_AC3"
SNP_FILE="$LD_DIR/SV_variants.txt"
ANNOT="$LD_DIR/annot_mat.txt"

# Run constrained analysis for all traits in sumstats/
for TRAIT in sumstats/*.sumstats.gz; do
    BASENAME=$(basename "$TRAIT" .sumstats.gz)
    OUT="test_results_constrained/univar_${BASENAME}.txt"
    OUT_LIST="test_results_constrained/univar_${BASENAME}_list.txt"

    # Skip if output already exists
    if [[ -f "$OUT_LIST" && -s "$OUT_LIST" ]]; then
        echo "Skipping (already exists): $OUT_LIST"
        continue
    fi

    echo "Processing (constrained): $BASENAME"

    python mixer_sv.py univar \
        --annot    "$ANNOT"    \
        --ld-mat1  "$LD_DIR"   \
        --trait1   "$TRAIT"    \
        --snp-file "$SNP_FILE" \
        --output   "$OUT"      \
        --constrain-roi-estimates-to-zero \
        --seed 42
done
```


---

## 6. CLI Arguments

All arguments below apply to the `univar` subcommand. Run `python mixer_sv.py univar --help` for the full help message.

### Required arguments

| Argument | Description |
|---|---|
| `--annot` | Path to the annotation matrix file (tab-separated, SNP-indexed) |
| `--ld-mat1` | Path to the LD reference directory containing `chr1.ldmat/` through `chr22.ldmat/` subdirectories |
| `--trait1` | Path to the GWAS summary statistics file (`.sumstats.gz`) |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--snp-file` | `None` | File listing SNP IDs that define the genomic region of interest (e.g., SV loci) |
| `--output`, `-o` | `output.txt` | Base path for output files (`_list.txt`, `_table.txt`, `.log` suffixes are added automatically) |
| `--maf-threshold` | `0.05` | Minor allele frequency threshold for variant filtering |
| `--seed` | `None` | Random seed for reproducibility |
| `--s-value` | `-0.25` | Heritability model parameter *S* (variants contribute via H^S) |
| `--pytorch-epochs` | `500` | Number of PyTorch optimization epochs |
| `--pytorch-lr` | `0.001` | PyTorch optimizer learning rate |
| `--only-base` | `False` | Use only the "base" (intercept) annotation column |
| `--disable-inverse-ld-score-weights` | `False` | Disable inverse-LD-score weighting |
| `--save-null-model` | `False` | Cache the null (baseline) model to a `.sig2_beta_i.mat` file for reuse across runs |
| `--constrain-roi-estimates-to-zero` | `False` | Constrain the within-region (SV) heritability coefficient to zero; useful as a null/constrained model for comparison against the unconstrained fit |
| `--verbose`, `-v` | `False` | Enable verbose logging |
| `--debug` | `False` | Enable debug-level logging |

---

## 7. Output Files

For a given `--output results/univar_trait.txt`, MiXeR-SV produces three files:

| File | Description |
|---|---|
| `results/univar_trait_list.txt` | Key-value pairs of all result metrics, one per line |
| `results/univar_trait_table.txt` | Tab-separated table with column headers (suitable for downstream aggregation across traits) |
| `results/univar_trait.log` | Full run log |

Result metrics include fold-enrichment estimates, standard errors, statistics, and per-annotation heritability contributions.

---

## 8. Citation

If you use MiXeR-SV in your research, please cite:

> Nguyen, D.T., Shadrin, A.A., Parker, N., Fuhrer, J., Vo, N.S., Dale, A.M., Andreassen, O.A., Frei, O. Structural variants contribute substantially to complex trait heritability. *bioRxiv* 2026.02.28.708732; doi: https://doi.org/10.64898/2026.02.28.708732 (to appear in *Genome Biology*).

---

## 9. License

MiXeR-SV is released under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).

---

## 10. Changelog

### v0.1
- First peer-reviewed release.
- Minor updates: license, LD reference URLs, renamed scripts, updated README.

### v0.0.1
- Initial pre-release: SV enrichment analysis.

