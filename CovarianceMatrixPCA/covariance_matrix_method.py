import sys
import time
import gc
import logging
import logging.config
import os, pdb

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import plotly_express as px
import plotly
import cvxpy as cvx
from distutils.util import strtobool
from file_processing import get_masked_matrix, process_labels_weights, logger_config
from scipy.spatial.distance import squareform, pdist
from skbio.stats.distance import mantel


def cov(x):
    ddof = 1
    x = np.ma.array(x, ndmin=2, copy=True, dtype=np.float32)
    xmask = np.ma.getmaskarray(x)
    rowvar = 1
    axis = 1 - rowvar
    xnotmask = np.logical_not(xmask).astype(np.float32) # why float32 and not bool?
    # Because if it is bool, 'np.dot(xnotmask, xnotmask.T)' in the next line would do boolean addition instead of float addition. So 'fact' would have 0 / 1 values instead of the number of unmasked entries used for computation of covariance for each pair of individuals. 
    fact = np.dot(xnotmask, xnotmask.T) * 1. - ddof
    del(xnotmask)
    gc.collect()
    result = (np.ma.dot(x, x.T, strict=False) / fact).squeeze()
    return result

def compute_strength_vector(X):
    strength_vector = np.sum(~np.isnan(X), axis=1) / X.shape[1]
    return strength_vector

def compute_strength_matrix(X):
    notmask = (~np.isnan(X)).astype(np.float32)
    strength_matrix = np.dot(notmask, notmask.T)
    strength_matrix /= X.shape[1]
    return strength_matrix

def create_validation_mask(X_incomplete, percent_inds):
    masked_rows = np.isnan(X_incomplete).any(axis=1)
    masked_inds = np.flatnonzero(masked_rows)
    X_masked = X_incomplete[masked_rows]
    percent_masked = 100 * np.isnan(X_masked).sum() / (X_masked.shape[0] * X_masked.shape[1])
    unmasked_rows = ~masked_rows
    X_unmasked = X_incomplete[unmasked_rows]
    masked_rows = np.random.choice(range(X_unmasked.shape[0]), size=int(X_unmasked.shape[0] * percent_inds / 100), replace=False)
    X_masked_rows = X_unmasked[masked_rows,:]
    mask = np.zeros(X_masked_rows.shape[0] * X_masked_rows.shape[1], dtype=np.int8)
    mask[:int(X_masked_rows.shape[0] * X_masked_rows.shape[1] * percent_masked / 100)] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool)
    mask = mask.reshape(X_masked_rows.shape)
    X_masked_rows[mask] = np.nan
    X_unmasked[masked_rows] = X_masked_rows
    X_incomplete[unmasked_rows] = X_unmasked
    masked_rows_new = np.isnan(X_incomplete).any(axis=1)
    masked_inds_new = np.flatnonzero(masked_rows_new)
    masked_inds_val = sorted(list(set(masked_inds_new) - set(masked_inds)))
    return X_incomplete, masked_inds_val

def run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename, robust=False):
    start_time = time.time()
    X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
    S = cov(X_incomplete).data
    X_incomplete =  X_incomplete.data
    strength_vec = compute_strength_vector(X_incomplete)
    strength_mat = compute_strength_matrix(X_incomplete)
    percent_inds_val = 20 # Percent of unmasked individuals to be masked for cross-validation 
    X_incomplete, masked_inds_val = create_validation_mask(X_incomplete, percent_inds_val) # masked_inds_val is the list of indices of the individuals masked for validation
    X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
    S_prime = cov(X_incomplete).data
    X_incomplete =  X_incomplete.data
    strength_vec_prime = compute_strength_vector(X_incomplete)
    strength_mat_prime = compute_strength_matrix(X_incomplete)
    weights_normalized = weights / weights.sum()
    weighted_sum = np.matmul(weights_normalized, S)
    weighted_rowsum = weighted_sum.reshape((1, S.shape[0]))
    weighted_colsum = weighted_sum.reshape((S.shape[0], 1))
    weighted_totalsum = np.dot(weighted_sum, weights_normalized)
    S = S - weighted_rowsum - weighted_colsum + weighted_totalsum
    logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
    W = np.diag(weights)
    WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
    if robust: 
        pass
    if save_cov_matrix:
        np.save(cov_matrix_filename, S.data)
        if robust:
            base, ext = os.path.split(cov_matrix_filename)
            np.save(f"{base}_completed_{lam}.{ext}", S.data)
    return WSW, S, W

def project_weighted_matrix(WSW, S, W):
    svd = TruncatedSVD(2, algorithm="arpack")
    svd.fit(WSW)
    X_projected = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
    total_var = np.trace(S)
    pc1_percentvar = 100 * svd.singular_values_[0] / total_var
    pc2_percentvar = 100 * svd.singular_values_[1] / total_var
    return X_projected, pc1_percentvar, pc2_percentvar


def NN_matrix_completion(cov, w, lam, cov0=None, verbose=False):
    """ Nuclear norm matrix completion (as in Candes and Recht, 2009 for a PSD matrix)
        objective: lam ||X||_* + ||w^1/2 (cov - X)||_2^2  with X = PSD 

    Parameters
    ----------
    cov : n x n 
        Input cov matrix to be denoised

    w : n x n
        Weight matrix for confidence in elements of cov 

    lam : float
        Regularization parameter determining the rank of the solution. Higher lam -> lower rank

    cov0 : n x n, optional
        Starting solution

    verbose : bool, default = False
        verbosity level of cvxpy

    Returns:
    --------
    X : n x n
        Completed matrix
    """
    n = cov.shape[0]
    X = cvx.Variable(shape=(n, n), PSD=True)
    if cov0 is None:
        u, s, v = svds(cov, k=10)
        cov0 = u@np.diag(s)@u.T
    X.value = cov0
    obj = cvx.Minimize(lam * cvx.norm(X, "nuc") +
                       cvx.sum_squares(cvx.multiply(np.sqrt(w), cov - X)))
    prob = cvx.Problem(obj, [])
    sol = prob.solve(warm_start=True, solver=cvx.SCS, vebose=verbose)
    return X.value

def matrix_completin(G, lams=None,  verbose=False, ncpus=None, parallel=False):
    """ 
    objective: lam ||X||_* + ||w^1/2 (cov - X)||_2^2  with X = PSD 

    Parameters
    ----------
    cov : n x n 
        Input cov matrix to be denoised

    w : n x n
        Weight matrix for confidence in elements of cov 

    lam : float
        Regularization parameter determining the rank of the solution. Higher lam -> lower rank

    cov0 : n x n, optional
        Starting solution

    verbose : bool, default = False
        verbosity level of cvxpy

    Returns:
    --------
    X : n x n
        Completed matrix
    """


    
    pass 


def scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels):
    plot_df = pd.DataFrame()
    plot_df['x'] = X_projected[:,0]
    plot_df['y'] = X_projected[:,1]
    plot_df['Label'] = labels
    plot_df['ID'] = ind_IDs
    scatter = px.scatter(plot_df, x='x', y='y', color='Label', hover_name='ID', color_discrete_sequence=px.colors.qualitative.Alphabet)
    plotly.offline.plot(scatter, filename = scatterplot_filename, auto_open=False)
    plot_df.to_csv(output_filename, columns=['ID', 'x', 'y'], sep='\t', index=False)
    
def run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_cov_matrix, cov_matrix_filename, robust):
    X_incomplete, ind_IDs, rs_IDs = get_masked_matrix(beagle_filename, vcf_filename, beagle_or_vcf, is_masked, vit_filename, fbk_filename, tsv_filename, vit_or_fbk_or_tsv, fb_or_msp, num_ancestries, ancestry, average_parents, prob_thresh)
    X_incomplete, ind_IDs, labels, weights = process_labels_weights(labels_filename, X_incomplete, ind_IDs, average_parents, is_weighted, save_masked_matrix, masked_matrix_filename)
    WSW, S, W = run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename, robust)
    if not robust:
        X_projected, pc1_percentvar, pc2_percentvar = project_weighted_matrix(WSW, S, W)
    else:
        pass

    scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels)
    print("Percent variance explained by the 1st principal component: ", pc1_percentvar)
    print("Percent variance explained by the 2nd principal component: ", pc2_percentvar)


def get_args(params_file):
    all_args = set(['BEAGLE_OR_VCF', 'BEAGLE_FILE', 'VCF_FILE', 'IS_MASKED',
                    'VIT_OR_FBK_OR_TSV', 'VIT_FILE', 'FBK_FILE', 'FB_OR_MSP', 'TSV_FILE',
                    'NUM_ANCESTRIES', 'ANCESTRY', 'PROB_THRESH', 'AVERAGE_PARENTS', 
                    'IS_WEIGHTED', 'LABELS_FILE', 'OUTPUT_FILE', 'SCATTERPLOT_FILE', 
                    'SAVE_MASKED_MATRIX', 'SAVE_MASKED_MATRIX', 'MASKED_MATRIX_FILE', 
                    'SAVE_COVARIANCE_MATRIX', 'COVARIANCE_MATRIX_FILE', 'ROBUST'])
    int_args = ['BEAGLE_OR_VCF', 'VIT_OR_FBK_OR_TSV', 'FB_OR_MSP', 'NUM_ANCESTRIES',
                'ANCESTRY']
    bool_args = ['IS_MASKED', 'AVERAGE_PARENTS', 'IS_WEIGHTED', 'SAVE_MASKED_MATRIX', 
            'SAVE_COVARIANCE_MATRIX', 'ROBUST']
    args = {}
    with open(params_file) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("#"):
                continue
            key_value = line.split('=')
            if len(key_value) == 2:
                key, value = key_value[0].strip(), key_value[1].strip()
                if key not in all_args:
                    logging.warning(f'parameter {key} is not an allowed parameter. Ignored!')
                else:
                    args[key] = value
            else:
                logging.warning(f'Line "{line}" does not follow the param=value format. Ignored!')
    keys = set(args.keys())
    remaining_keys = all_args - keys
    if remaining_keys:
        logging.error(f'Please specify all required keys! {remaining_keys} are missing')
        print(f'Please specify all required keys! {remaining_keys} are missing')
        raise ValueError
    for val in int_args:
        args[val] = int(args[val])
    for val in bool_args:
        args[val] = bool(strtobool(args[val]) )
    logging.debug(f"parameters used are: {args}")
    return args


def run(params_filename):
    args = get_args(params_filename)
    kwargs = {key.lower().replace("covariance", "cov").replace("file", "filename"): 
            args[key] for key in args.keys()}
#    beagle_or_vcf = int(params['BEAGLE_OR_VCF'])
#    beagle_filename = str(params['BEAGLE_FILE'])
#    vcf_filename = str(params['VCF_FILE'])
#    is_masked = bool(strtobool(params['IS_MASKED']))
#    vit_or_fbk_or_tsv = int(params['VIT_OR_FBK_OR_TSV'])
#    vit_filename = str(params['VIT_FILE'])
#    fbk_filename = str(params['FBK_FILE'])
#    fb_or_msp = int(params['FB_OR_MSP'])
#    tsv_filename = str(params['TSV_FILE'])
#    num_ancestries = int(params['NUM_ANCESTRIES'])
#    ancestry = int(params['ANCESTRY'])
#    prob_thresh = float(params['PROB_THRESH'])
#    average_parents = bool(strtobool(params['AVERAGE_PARENTS']))
#    is_weighted = bool(strtobool(params['IS_WEIGHTED']))
#    labels_filename = str(params['LABELS_FILE'])
#    output_filename = str(params['OUTPUT_FILE'])
#    scatterplot_filename = str(params['SCATTERPLOT_FILE'])
#    save_masked_matrix = bool(strtobool(params['SAVE_MASKED_MATRIX']))
#    masked_matrix_filename = str(params['MASKED_MATRIX_FILE'])
#    save_cov_matrix = bool(strtobool(params['SAVE_COVARIANCE_MATRIX']))
#    cov_matrix_filename = str(params['COVARIANCE_MATRIX_FILE'])
#    run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_cov_matrix, cov_matrix_filename)
    run_method(**kwargs)


def main():
    logging.config.dictConfig(logger_config(True))
    logging.info("Total time ---  seconds ---")
    params_filename = sys.argv[1]
    start_time = time.time()
    run(params_filename)
    logging.info("Total time --- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()
