"""
Sumstat QC utilities for flexible processing of GWAS summary statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict
import os
import gzip
from scipy.io import loadmat

COLUMN_MAPPINGS = {
    'snp': ['SNP', 'snp', 'ID', 'id', 'rsid', 'RSID', 'MarkerName', 'variant_id'],
    'chr': ['CHR', 'chr', 'chromosome', 'CHROMOSOME', '#CHROM'],
    'pos': ['POS', 'pos', 'BP', 'bp', 'position', 'POSITION'],
    'a1': ['A1', 'a1', 'ALT', 'alt', 'effect_allele', 'EA'],#rename to effect_allele, remove A1 / a1 from the default
    'a2': ['A2', 'a2', 'REF', 'ref', 'other_allele', 'NEA'], #rename other_allele
    'beta': ['BETA', 'beta', 'b', 'B', 'effect', 'EFFECT', 'logOR'],
    'se': ['SE', 'se', 'standard_error'],
    'z': ['Z', 'z', 'zscore', 'ZSCORE'],
    'n': ['N', 'n', 'NMISS', 'n_samples', 'sample_size'],
    #'info': ['INFO'],
    #'direction': ['DIRECTION'],
    #'nca': ['NCASES'],
    #'nco': ['NCONTROLS'],
    #'or': ['OR']
}


class SumstatQC:
    """
    Class for quality control and processing of GWAS summary statistics.

    Features:
    - Flexible column name mapping
    - Merge with reference info
    - Filter by MAF and MHC region
    - Remove ambiguous SNPs
    - Align alleles and flip effect sizes/z-scores
    """

    def __init__(self, sumstat_df: pd.DataFrame, info_df: pd.DataFrame):
        """
        Initialize SumstatQC by standardizing columns, validating, and merging with reference info.

        Args:
            sumstat_df: GWAS summary statistics dataframe.
            info_df: Reference annotation/info dataframe with required columns:
                     ['chr', 'pos', 'snp', 'a1', 'a2', 'a1f', 'maf', 'het'].
        """
        sumstat_df = self._standardize_columns(sumstat_df.copy())
        if 'chr' in sumstat_df.columns:
            sumstat_df = self._standardize_chr(sumstat_df)
        sumstat_df = self._validate_sumstat(sumstat_df)

        self._validate_info_columns(info_df)
        info_df = self._standardize_columns(info_df.copy())
        info_df = self._standardize_chr(info_df)

        merged = info_df.merge(sumstat_df, on='snp', suffixes=('_ref', '_sum'), how='inner')
        merged = merged.rename(columns={'a1_ref': 'a1', 'a2_ref': 'a2'})

        # Normalize alleles to uppercase
        for c in ['a1', 'a2', 'a1_sum', 'a2_sum']:
            merged[c] = merged[c].str.upper()

        self.sumstat = sumstat_df
        self.info = info_df
        self.merged = merged

    # ---------------- Private helper methods ----------------

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map common column name variants to standard names.

        Args:
            df: DataFrame to standardize.

        Returns:
            DataFrame with standardized column names.
        """
        rename_map = {}
        for std, variants in COLUMN_MAPPINGS.items():
            for c in df.columns:
                if c in variants:
                    rename_map[c] = std
                    break
        return df.rename(columns=rename_map)

    @staticmethod
    def _validate_info_columns(info: pd.DataFrame):
        """
        Check that reference info contains required columns.

        Args:
            info: Reference info dataframe.

        Raises:
            ValueError if any required column is missing.
        """
        required = ['chr', 'pos', 'snp', 'a1', 'a2', 'a1f', 'maf', 'het']
        missing = [c for c in required if c not in info.columns]
        if missing:
            raise ValueError(f"Reference info missing columns: {missing}")

    @staticmethod
    def _standardize_chr(df: pd.DataFrame):
        """
        Ensure chromosome column is string without 'chr' prefix.

        Args:
            df: DataFrame with 'chr' column.

        Returns:
            DataFrame with standardized 'chr' column.
        """
        df['chr'] = df['chr'].astype(str).str.removeprefix('chr').str.removeprefix('CHR')
        return df

    @staticmethod
    def _validate_sumstat(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean GWAS summary statistics.

        Actions:
        - Compute z-score if missing and beta/se are present.
        - Generate 'snp' ID from chr_pos if missing.
        - Keep only relevant columns ['snp','a1','a2','n','z','beta','se'].
        - Remove rows with missing or non-positive sample size.
        - Ensure minimum required columns exist: ['snp','a1','a2','n','z'].

        Args:
            df: Summary statistics dataframe.

        Returns:
            Validated and filtered dataframe.

        Raises:
            ValueError if required data is missing.
        """
        # Compute z if missing
        if 'z' not in df.columns:
            if {'beta', 'se'}.issubset(df.columns):
                df['z'] = df['beta'] / df['se']
            else:
                raise ValueError("Missing z or (beta, se).")

        # Generate snp ID if missing
        if 'snp' not in df.columns:
            if {'chr', 'pos'}.issubset(df.columns):
                df['snp'] = df['chr'].astype(str) + '_' + df['pos'].astype(str)
            else:
                raise ValueError("Missing snp or (chr, pos).")

        # Keep only relevant columns if they exist
        required_cols = ['snp', 'a1', 'a2', 'n', 'z', 'beta', 'se']
        df = df[[c for c in required_cols if c in df.columns]]

        # Drop rows with missing or non-positive sample size
        if 'n' in df.columns:
            df = df[df['n'].notna() & (df['n'] > 0)]

        # Ensure minimum required columns exist
        min_required = ['snp', 'a1', 'a2', 'n', 'z']
        missing_min = [c for c in min_required if c not in df.columns]
        if missing_min:
            raise ValueError(f"Sumstat is missing required columns: {missing_min}")

        return df

    # ---------------- Public QC methods ----------------

    def filter_maf(self, maf_thresh: float = 0.05):
        """
        Filter SNPs by minor allele frequency (MAF).

        Args:
            maf_thresh: Keep SNPs with maf >= maf_thresh and <= 1-maf_thresh.

        Returns:
            Self, for method chaining.
        """
        m = (self.merged['maf'] >= maf_thresh) & (self.merged['maf'] <= 1 - maf_thresh)
        self.merged = self.merged.loc[m].copy()
        return self

    def remove_ambiguous(self):
        """
        Remove strand-ambiguous SNPs (A/T, T/A, C/G, G/C).

        Returns:
            Self, for method chaining.
        """
        ambig = {'AT', 'TA', 'CG', 'GC'}
        pair = self.merged['a1_sum'] + self.merged['a2_sum']
        self.merged = self.merged.loc[~pair.isin(ambig)].copy()
        return self

    def filter_mhc(self, mhc_chr='6', mhc_start=25_000_000, mhc_end=34_000_000):
        """
        Remove SNPs in the MHC region.

        Args:
            mhc_chr: Chromosome containing MHC region.
            mhc_start: Start position of MHC region.
            mhc_end: End position of MHC region.

        Returns:
            Self, for method chaining.
        """
        mhc_chr = str(mhc_chr).removeprefix('chr')
        m = (
            (self.merged['chr'] == mhc_chr) &
            (self.merged['pos'].between(mhc_start, mhc_end))
        )
        self.merged = self.merged.loc[~m].copy()
        return self

    def align_alleles(self):
        """
        Align alleles to reference and flip signed effects if needed.

        Keeps original 'z' and 'beta', stores aligned versions as:
            - 'z_aligned' (always)
            - 'beta_aligned' (if beta exists)

        Drops SNPs with incompatible alleles.

        Returns:
            Self, for method chaining.
        """
        a1s, a2s = self.merged['a1_sum'], self.merged['a2_sum']
        a1r, a2r = self.merged['a1'], self.merged['a2']

        # Determine matches and flips
        match = (a1s == a1r) & (a2s == a2r)
        flip  = (a1s == a2r) & (a2s == a1r)

        # match from other standr
        #https://github.com/precimed/mixer_matlab/blob/main/sumstats/sumstats2mat.py#L21
        # ref A,G.  sum T,C

        # Keep only compatible SNPs
        self.merged = self.merged.loc[match | flip].copy()

        # Store aligned z
        self.merged['z_aligned'] = self.merged['z'].copy()
        self.merged.loc[flip, 'z_aligned'] *= -1

        # Store beta_aligned only if beta exists
        if 'beta' in self.merged.columns:
            self.merged['beta_aligned'] = self.merged['beta'].copy()
            self.merged.loc[flip, 'beta_aligned'] *= -1

        return self

    # ---------------- Accessor methods ----------------

    def get_filtered(self) -> pd.DataFrame:
        """
        Return the QC-filtered and merged dataframe.
        """
        return self.merged

    def to_file(self, path: str, sep='\t', **kwargs):
        """
        Save QC-filtered dataframe to file.

        Args:
            path: File path to save.
            sep: Column separator (default tab).
            **kwargs: Additional arguments passed to pd.DataFrame.to_csv.

        Returns:
            Self, for method chaining.
        """
        self.merged.to_csv(path, sep=sep, index=False, **kwargs)
        return self

    def summary(self) -> Dict:
        """
        Return a summary of QC results.

        Returns:
            dict with keys:
                n_sumstat: Number of input summary SNPs.
                n_ref: Number of reference SNPs.
                n_filtered: Number of SNPs after filtering.
                filter_rate: Fraction retained.
        """
        return {
            'n_sumstat': len(self.sumstat),
            'n_ref': len(self.info),
            'n_filtered': len(self.merged),
            'filter_rate': len(self.merged) / len(self.sumstat) if len(self.sumstat) else 0
        }

    def get_defvec(self) -> np.ndarray:
        """
        Get a boolean indicator vector showing which reference SNPs are retained
        in the filtered merged data.

        Returns:
            np.ndarray of shape (n_ref,), where True indicates the reference SNP
            is present in `self.merged` after QC, False otherwise.
        """
        return self.info['snp'].isin(self.merged['snp']).to_numpy()

    def get_nvec(self) -> np.ndarray:
        """
        Get an array of sample sizes for all reference SNPs, using n from merged if present, else np.nan.

        Returns:
            np.ndarray of sample sizes, shape (n_ref,), aligned to self.info['snp'] order.
        """
        result = self.info[['snp']].merge(
            self.merged[['snp', 'n']], on='snp', how='left'
        )
        # Handle duplicates in merged data to ensure output matches self.info length
        if len(result) != len(self.info):
            result = result.drop_duplicates('snp', keep='first')
        nvec = result['n'].to_numpy()
        return nvec
        

    def get_zvec(self) -> np.ndarray:
        """
        Get an array of aligned z-scores for all reference SNPs, using z_aligned from merged if present, else np.nan.

        Returns:
            np.ndarray of z-scores, shape (n_ref,), aligned to self.info['snp'] order.
        """
        result = self.info[['snp']].merge(
            self.merged[['snp', 'z_aligned']], on='snp', how='left'
        )
        # Handle duplicates in merged data to ensure output matches self.info length
        if len(result) != len(self.info):
            result = result.drop_duplicates('snp', keep='first')
        zvec = result['z_aligned'].to_numpy()
        return zvec

    def __repr__(self):
        return (
            f"SumstatQC(n_sumstat={len(self.sumstat)}, "
            f"n_ref={len(self.info)}, n_filtered={len(self.merged)})"
        )


# ---------------- Helper functions ----------------

def load_sumstat(filepath: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()

    # MATLAB .mat file
    if ext == ".mat":
        mat = loadmat(filepath)
        data = {k: v.squeeze() for k, v in mat.items() if not k.startswith("__")}
        return pd.DataFrame(data)

    # CSV → use fast C engine
    if ext == ".csv":
        return pd.read_csv(filepath, **kwargs)

    # All other text / gzipped text → safe whitespace parsing
    if "sep" not in kwargs:
        kwargs["sep"] = r"\s+"
        kwargs["engine"] = "python"

    return pd.read_csv(filepath, **kwargs)




def load_info(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load reference annotation/info file.

    Args:
        filepath: Path to reference info file.
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        pd.DataFrame with reference info.
    """
    if 'sep' not in kwargs:
        kwargs['sep'] = '\t'
    return pd.read_csv(filepath, **kwargs)
