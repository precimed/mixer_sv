#!/usr/bin/env python3

"""
MIXER Tool for Structural Variant Enrichment Analysis with Integrated PyTorch Optimization
"""

import argparse
import numpy as np
import pandas as pd
import random
import torch
import copy
import logging
import utils.my_logging
import os
import scipy.io as sio
from scipy.sparse import issparse, diags

#from utils.data import load_sparse_mat, load_ldsc_data, load_trait, find_weights
from utils.utils import find_fold_enrich, find_fold_enrich_se
from utils.stats import likelihood_ratio_test
from utils.optim import train_mixer_univariate_MoM, train_mixer_bivariate_MoM
from utils.optim_mle import train_mixer_univariate_MLE, train_mixer_bivariate_MLE
from utils.hessian_se import find_univariate_se, find_bivariate_se
from utils.ld_matrix import LDMatrix
from utils.sumstat import load_sumstat, SumstatQC

logger = logging.getLogger(__name__)

# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================

def read_annot(annot_file):
    """Read annotation file and return as dataframe indexed by SNP"""
    df = pd.read_csv(annot_file, sep='\t')
    df = df.set_index('snp')
    return df

def load_annotations(args):
    """Load genome-wide annotations"""
    logger.info("Loading baseline annotations...")
    base_annot = read_annot(args.annot)
    annomat_BASE = base_annot.values
    annonames_BASE = list(base_annot.columns)

    if args.only_base:
        annomat_BASE = annomat_BASE[:, [0]]
        annonames_BASE = [annonames_BASE[0]]
    
    # Load ROI SNP list if provided
    roi_snps = None
    if args.snp_file:
        roi_snps = set(pd.read_csv(args.snp_file, header=None)[0].values)
    
    return base_annot, annomat_BASE, annonames_BASE, roi_snps


def process_chromosome(chr_num, ld_dir, base_annot, roi_snps, sumstat_file1, sumstat_file2, args):
    """Process a single chromosome and return accumulated LD-weighted matrices
    
    Returns:
        dict with keys: zvec, nvec, hetvec, defvec, weights, annomat_BASE_ld, annomat_ROI_ld (and trait2 versions if bivar)
    """
    # Load sumstats first to avoid unnecessary LD loading if errors
    ss1 = load_sumstat(sumstat_file1)

    # load chromosome LD directory
    chr_dir = os.path.join(ld_dir, f"chr{chr_num}.ldmat")
    if not os.path.exists(chr_dir):
        logger.warning(f"Chromosome {chr_num} directory not found: {chr_dir}")
        return None
    
    logger.info(f"Processing chromosome {chr_num}...")
    
    # Load chromosome-specific LD and info
    ld_obj = LDMatrix.load_data(chr_dir)
    info = ld_obj.info.copy()
    n_snps_chr = len(info)
    
    # Filter annotations to this chromosome
    chr_annot_mask = base_annot.index.isin(info['snp'])
    if not chr_annot_mask.any():
        logger.warning(f"No annotation overlap for chromosome {chr_num}, skipping")
        return None
    
    annomat_BASE_chr = base_annot.loc[chr_annot_mask].values
    
    # Create ROI annotation for this chromosome
    if roi_snps is not None:
        defvec_roi = info['snp'].isin(roi_snps).values.astype(int)
        defvec_other = (~info['snp'].isin(roi_snps)).values.astype(int)
        defvec_all = np.ones(n_snps_chr, dtype=int)
        annomat_ROI_chr = np.column_stack([defvec_all, defvec_roi, defvec_other])
    else:
        defvec_all = np.ones(n_snps_chr, dtype=int)
        annomat_ROI_chr = np.column_stack([defvec_all])
    
    # QC sumstats for this chromosome
    ss1_qc = SumstatQC(ss1, info)
    ss1_qc.remove_ambiguous().filter_maf(args.maf_threshold).filter_mhc().align_alleles()
    
    zvec1 = ss1_qc.get_zvec()
    nvec1 = ss1_qc.get_nvec()
    hetvec1 = ss1_qc.info.het.values
    defvec1 = ss1_qc.get_defvec()
    
    # Compute LD scores and weights
    LD_r_sq = ld_obj.get_full_ld_matrix_sq()
    w_ld = LD_r_sq.dot(defvec1.astype(int))
    weights1 = np.zeros(n_snps_chr, dtype=float)
    weights1[defvec1] = 1 / w_ld[defvec1]
    defvec1 = defvec1 & (weights1 > 0)
    
    result = {
        'chr': chr_num,
        'n_snps': n_snps_chr,
        'n_snps_qc': np.sum(defvec1),
        'zvec1': zvec1,
        'nvec1': nvec1,
        'hetvec1': hetvec1,
        'defvec1': defvec1,
        'weights1': weights1,
        'annomat_BASE_chr': annomat_BASE_chr,
        'annomat_ROI_chr': annomat_ROI_chr,
        'LD_r_sq': LD_r_sq
    }
    
    # If bivariate, process trait 2
    if sumstat_file2 is not None:
        ss2 = load_sumstat(sumstat_file2)
        ss2_qc = SumstatQC(ss2, info)
        ss2_qc.remove_ambiguous().filter_maf(args.maf_threshold).filter_mhc().align_alleles()
        
        zvec2 = ss2_qc.get_zvec()
        nvec2 = ss2_qc.get_nvec()
        hetvec2 = ss2_qc.info.het.values
        defvec2 = ss2_qc.get_defvec()
        
        w_ld2 = LD_r_sq.dot(defvec2.astype(int))
        weights2 = np.zeros(n_snps_chr, dtype=float)
        weights2[defvec2] = 1 / w_ld2[defvec2]
        defvec2 = defvec2 & (weights2 > 0)
        
        result.update({
            'zvec2': zvec2,
            'nvec2': nvec2,
            'hetvec2': hetvec2,
            'defvec2': defvec2,
            'weights2': weights2,
            'defvec_both': defvec1 & defvec2
        })
    
    return result


def accumulate_chromosome_data(chr_results_list, args, is_bivar=False):
    """Accumulate data from all chromosomes into genome-wide arrays"""
    logger.info("Accumulating data across chromosomes...")
    
    # Concatenate arrays across chromosomes
    zvec1_list = [r['zvec1'] for r in chr_results_list]
    nvec1_list = [r['nvec1'] for r in chr_results_list]
    hetvec1_list = [r['hetvec1'] for r in chr_results_list]
    defvec1_list = [r['defvec1'] for r in chr_results_list]
    weights1_list = [r['weights1'] for r in chr_results_list]
    
    zvec1 = np.concatenate(zvec1_list)
    nvec1 = np.concatenate(nvec1_list)
    hetvec1 = np.concatenate(hetvec1_list)
    defvec1 = np.concatenate(defvec1_list)
    weights1 = np.concatenate(weights1_list)
    
    # Concatenate annotation matrices
    annomat_BASE_list = [r['annomat_BASE_chr'] for r in chr_results_list]
    annomat_ROI_list = [r['annomat_ROI_chr'] for r in chr_results_list]
    annomat_BASE = np.vstack(annomat_BASE_list)
    annomat_ROI = np.vstack(annomat_ROI_list)
    
    result = {
        'zvec1': zvec1,
        'nvec1': nvec1,
        'hetvec1': hetvec1,
        'defvec1': defvec1,
        'weights1': weights1,
        'annomat_BASE': annomat_BASE,
        'annomat_ROI': annomat_ROI,
        'n_snps_total': len(zvec1),
        'n_snps_qc': np.sum(defvec1)
    }
    
    if is_bivar:
        zvec2 = np.concatenate([r['zvec2'] for r in chr_results_list])
        nvec2 = np.concatenate([r['nvec2'] for r in chr_results_list])
        hetvec2 = np.concatenate([r['hetvec2'] for r in chr_results_list])
        defvec2 = np.concatenate([r['defvec2'] for r in chr_results_list])
        weights2 = np.concatenate([r['weights2'] for r in chr_results_list])
        defvec_both = np.concatenate([r['defvec_both'] for r in chr_results_list])
        
        result.update({
            'zvec2': zvec2,
            'nvec2': nvec2,
            'hetvec2': hetvec2,
            'defvec2': defvec2,
            'weights2': weights2,
            'defvec_both': defvec_both
        })
    
    logger.info(f"Total SNPs: {result['n_snps_total']:,}, QC passed: {result['n_snps_qc']:,}")
    
    return result

# =====================================================================================
# UNIVARIATE ANALYSIS
# =====================================================================================

def compute_ld_weighted_annot_univar(chr_results_list, sig2_beta_i_genome, annomat_BASE, annomat_ROI):
    """Compute LD-weighted annotation matrices by iterating through chromosomes
    
    Returns LD-weighted matrices accumulated across all chromosomes
    """
    logger.info("Computing LD-weighted annotation matrices across chromosomes...")
    
    n_snps_total = sum(r['n_snps'] for r in chr_results_list)
    n_base_annot = annomat_BASE.shape[1]
    n_roi_annot = annomat_ROI.shape[1]
    
    # Initialize accumulated matrices
    annomat_BASE_ld = np.zeros((n_snps_total, n_base_annot))
    annomat_ROI_ld = np.zeros((n_snps_total, n_roi_annot))
    
    offset = 0
    for r in chr_results_list:
        n_snps_chr = r['n_snps']
        sig2_beta_i_chr = sig2_beta_i_genome[offset:offset+n_snps_chr]
        
        # Compute LD-weighted matrices for this chromosome
        LD_r_sq = r['LD_r_sq']
        annomat_BASE_chr = r['annomat_BASE_chr']
        annomat_ROI_chr = r['annomat_ROI_chr']
        
        annomat_BASE_ld[offset:offset+n_snps_chr] = LD_r_sq.dot(diags(sig2_beta_i_chr)).dot(annomat_BASE_chr)
        annomat_ROI_ld[offset:offset+n_snps_chr] = LD_r_sq.dot(diags(sig2_beta_i_chr)).dot(annomat_ROI_chr)
        
        offset += n_snps_chr
        logger.info(f"  Chr {r['chr']}: processed {n_snps_chr:,} SNPs")
    
    return annomat_BASE_ld, annomat_ROI_ld


def univariate_analysis(args, data, chr_results_list, annomat_BASE, annonames_BASE, annomat_ROI):
    """Perform univariate analysis using chromosome-wise accumulated data"""
    
    zvec1 = data['zvec1']
    nvec1 = data['nvec1']
    hetvec = data['hetvec1']
    weights = data['weights1']
    
    if args.save_null_model and os.path.exists(args.trait1 + '.sig2_beta_i.mat'):
        logger.info("Reading in null model from <trait1>.sig2_beta_i.mat file...")
        sig2_beta_i = sio.loadmat(args.trait1 + '.sig2_beta_i.mat')['sig2_beta_i'].reshape((-1, ))
        if len(sig2_beta_i) != len(hetvec): 
            raise ValueError(f"{args.trait1 + '.sig2_beta_i.mat'} is incompatible with reference file.")
        fold_enrich_base_vs_null = None
    else:
        logger.info("Fitting null model (heritability enrichments across functional annotations)...")
        sig2_beta_i = np.power(hetvec, 1+args.s_value)
        
        # Compute LD-weighted BASE annotation matrices across chromosomes
        annomat_BASE_ld, _ = compute_ld_weighted_annot_univar(chr_results_list, sig2_beta_i, annomat_BASE, annomat_ROI)
        
        base_params = train_mixer_univariate_MoM(zvec1, nvec1, weights, annomat_BASE_ld)
        logger.info(f"\tBaseline params: {base_params}")

        fold_enrich_base_vs_null, _ = find_fold_enrich(
            annomat_BASE, sig2_beta_i, sig2_beta_i * (annomat_BASE @ base_params.coefs[1:])
        )

        sig2_beta_i = sig2_beta_i * (annomat_BASE @ base_params.coefs[1:])
        if args.save_null_model:
            logger.info("Saving null model (<trait1>.sig2_beta_i.mat file)...")
            sio.savemat(args.trait1 + '.sig2_beta_i.mat', 
                       {'sig2_beta_i': sig2_beta_i}, 
                       format='5', do_compression=False, oned_as='column', appendmat=False)

    # Compute LD-weighted ROI annotation matrices across chromosomes
    _, annomat_ROI_ld = compute_ld_weighted_annot_univar(chr_results_list, sig2_beta_i, annomat_BASE, annomat_ROI)

    logger.info("Fitting baseline model...")
    base_annot_idx = [0]
    base_params = train_mixer_univariate_MoM(zvec1, nvec1, weights, annomat_ROI_ld[:, base_annot_idx])
    logger.info(f"\tBaseline model params: {base_params}")

    if annomat_ROI.shape[1] == 1:
        logger.warning("Only 'base' annotation found in ROI annotation matrix; skipping full model fitting.")
        results = {
            "sig2beta": base_params.coefs[1],
            "sig2zero_trait1": base_params.coefs[0],
            "total_h2_trait1": np.sum(sig2_beta_i * (annomat_ROI[:, base_annot_idx] @ base_params.coefs[1:]))
        }
        return results

    logger.info("Fitting full model...")
    annot_idx = [2, 1]
    full_params = train_mixer_univariate_MoM(zvec1, nvec1, weights, annomat_ROI_ld[:, annot_idx])
    if args.constrain_roi_estimates_to_zero:
        full_params.effective_params_mask[2] = False
        full_params.coefs[2] = 0.0
    full_params = train_mixer_univariate_MLE(
        zvec1, nvec1, weights, annomat_ROI_ld[:, annot_idx], full_params,
        n_epochs=args.pytorch_epochs, lr=args.pytorch_lr
    )
    logger.info(f"\tFull model params: {full_params}")

    full_coefs1_se, full_coefs1_se_sample = find_univariate_se(
        full_params, zvec1, nvec1, weights, annomat_ROI_ld[:, annot_idx]
    )

    fold_enrich_ROI_vs_base, fold_enrich_ROI_vs_base_se, h2_total, h2_total_se, h2_regional, h2_regional_se = find_fold_enrich_se(
        annomat_ROI[:, [0] + annot_idx], 
        annomat_ROI[:, annot_idx], 
        sig2_beta_i, full_params.coefs, full_coefs1_se_sample
    )

    Change_logL_trait1 = full_params.logL - base_params.logL
    Delta_AIC_trait1 = base_params.AIC - full_params.AIC

    results = {
        "Change_logL_trait1": Change_logL_trait1,
        "Delta_AIC_trait1": Delta_AIC_trait1,
        "sig2beta_outside_region_trait1": full_params.coefs[1],
        "sig2beta_outside_region_trait1_se": full_coefs1_se[1],
        "sig2beta_within_region_trait1": full_params.coefs[2],
        "sig2beta_within_region_trait1_se": full_coefs1_se[2],
        "enrich_vs_base_outside_region_trait1": fold_enrich_ROI_vs_base[1],
        "enrich_vs_base_outside_region_trait1_se": fold_enrich_ROI_vs_base_se[1],
        "enrich_vs_base_within_region_trait1": fold_enrich_ROI_vs_base[2],
        "enrich_vs_base_within_region_trait1_se": fold_enrich_ROI_vs_base_se[2],
        "sig2zero_trait1": full_params.coefs[0],
        "sig2zero_trait1_se": full_coefs1_se[0],
        "total_h2_trait1": h2_total,
        "total_h2_trait1_se": h2_total_se,
        "h2_outside_region_trait1": h2_regional[1],
        "h2_outside_region_trait1_se": h2_regional_se[1],
        "h2_within_region_trait1": h2_regional[2],
        "h2_within_region_trait1_se": h2_regional_se[2]
    }

    if fold_enrich_base_vs_null is not None:
        for annoname, fold_enrich in zip(annonames_BASE, fold_enrich_base_vs_null):
            results[f'enrich_vs_null_{annoname}_trait1'] = fold_enrich

    return results


# =====================================================================================
# SUBCOMMAND HANDLERS
# =====================================================================================

def run_univar(args):
    """Handler for univar subcommand"""
    logger.info("=== UNIVARIATE ANALYSIS ===")

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load annotations
    base_annot, annomat_BASE, annonames_BASE, roi_snps = load_annotations(args)
    
    # Process chromosomes 1-22
    chr_results_list = []
    for chr_num in range(1, 23):
        result = process_chromosome(
            chr_num, args.ld_mat1, base_annot, roi_snps, 
            args.trait1, None, args
        )
        if result is not None:
            chr_results_list.append(result)
    
    if not chr_results_list:
        raise ValueError("No chromosomes were successfully processed")
    
    # Accumulate data across chromosomes
    data = accumulate_chromosome_data(chr_results_list, args, is_bivar=False)
    
    # Run univariate analysis
    results = univariate_analysis(args, data, chr_results_list, annomat_BASE, annonames_BASE, data['annomat_ROI'])

    write_results(args, results)


def write_results(args, results):
    """Write results to output files"""
    output_base_filename = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
    list_output = f"{output_base_filename}_list.txt"
    table_output = f"{output_base_filename}_table.txt"

    with open(list_output, "w") as f:
        for key, value in results.items():
            line = f"{key}: {value}\n"
            f.write(line)
            logger.info(line.strip())

    with open(table_output, "w") as f:
        headers = list(results.keys())
        headers.sort()
        f.write('\t'.join(headers) + '\n')
        row_values = [str(results[header]) for header in headers]
        f.write('\t'.join(row_values) + '\n')

    logger.info(f"Analysis complete.")
    logger.info(f"List results: {list_output}")
    logger.info(f"Table results: {table_output}")

# =====================================================================================
# ARGUMENT PARSERS
# =====================================================================================

def add_common_arguments(parser):
    """Add arguments common to all subcommands"""
    parser.add_argument('--annot', required=True,
                       help='Path to annotation matrix file')
    parser.add_argument('--ld-mat1', required=True,
                       help='Path to parent directory containing chr1.ldmat/ through chr22.ldmat/ subdirectories')
    parser.add_argument('--trait1', required=True,
                       help='Path to trait 1 summary statistics file')
    parser.add_argument('--snp-file', required=False,
                       help='Path to SNP list file defining the genomic region of interest')
    parser.add_argument('--output', '-o', default='output.txt',
                       help='Output file name (default: output.txt)')
    parser.add_argument('--maf-threshold', type=float, default=0.005,
                       help='MAF threshold for filtering (default: 0.005)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug output')
    parser.add_argument('--disable-inverse-ld-score-weights', default=False, action="store_true",
                       help="Disable weighting by inverse LD score")
    parser.add_argument('--only-base', default=False, action='store_true',
                       help='Use only "base" from annotation file')
    parser.add_argument('--s-value', default=-0.25, type=float,
                       help="'S' parameter of heritability model, contributing via H^S (default: %(default)s)")
    parser.add_argument('--save-null-model', default=False, action='store_true',
                       help='Save/reuse baseline model across runs via .sig2_beta_i.mat file')
    parser.add_argument('--constrain-roi-estimates-to-zero', default=False, action='store_true',
                       help='Constrain ROI estimates to zero (default: False)')
    parser.add_argument('--pytorch-epochs', type=int, default=500,
                       help='Number of PyTorch optimization epochs (default: 500)')
    parser.add_argument('--pytorch-lr', type=float, default=0.001,
                       help='PyTorch learning rate (default: 0.001)')

def main():
    parser = argparse.ArgumentParser(
        description='MIXER Tool for Structural Variant Enrichment Analysis with Integrated PyTorch Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Univariate analysis
            python mixer_sv.py univar \
                --annot data/reference/annot_mat.txt \
                --ld-mat1 /path/to/ld_reference/ \\
                --trait1 data/sumstats/trait1.sumstats \\
                --snp-file data/regions/genes.txt \\
                --output results/univar_output.txt \\
                --seed 42

            """
    )

    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    subparsers.required = True

    # Univariate subcommand
    parser_univar = subparsers.add_parser('univar', 
                                           help='Perform univariate analysis',
                                           formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser_univar)
    parser_univar.set_defaults(func=run_univar)

    args = parser.parse_args()

    output_base_filename = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
    utils.my_logging.setup_logger(f'{output_base_filename}.log', args.verbose)

    args.func(args)

if __name__ == "__main__":
    main()
