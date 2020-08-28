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
from scipy.sparse.linalg import svds
from distutils.util import strtobool
from gen_tools import array_process, process_labels_weights, logger_config
from scipy.spatial.distance import squareform, pdist
from skbio.stats.distance import mantel
import scipy
from sklearn.utils import check_array
from ast import literal_eval

def process_masks(masks, rs_ID_list, ind_ID_list, ancestry, is_masked):
    masked_matrix = masks[0][ancestry].T
    rs_IDs = rs_ID_list[0]
    ind_IDs = ind_ID_list[0]
    return masked_matrix, rs_IDs, ind_IDs

def load_mask_file(masks_file):
    mask_files = np.load(masks_file, allow_pickle=True)
    masks = mask_files['masks']
    rs_ID_list = mask_files['rs_ID_list']
    ind_ID_list = mask_files['ind_ID_list']
    labels = mask_files['labels']
    weights = mask_files['weights']
    return masks, rs_ID_list, ind_ID_list, labels, weights

def cov(x):
    ddof = 1
    x = np.ma.array(x, ndmin=2, copy=True, dtype=np.float32)
    xmask = np.ma.getmaskarray(x)
    rowvar = 1
    axis = 1 - rowvar
    xnotmask = np.logical_not(xmask).astype(np.float32) 
    fact = np.dot(xnotmask, xnotmask.T) * 1. - ddof
    del(xnotmask)
    gc.collect()
    result = (np.ma.dot(x, x.T, strict=False) / fact).squeeze()
    x =  x.data
    strength_vec = compute_strength_vector(x)
    strength_mat = compute_strength_matrix(x)
    return result.data, strength_vec, strength_mat

#@profile
def direct_dot(A):
    if A.flags.c_contiguous:
        A = check_array(A, copy=False, order="F")
    dot = scipy.linalg.get_blas_funcs("syrk", (A,))
    m = dot(alpha=1.0, a=A)
    m += m.T 
    m -= np.diag(np.diag(A))
    return m


def compute_strength_vector(X):
    strength_vector = np.sum(~np.isnan(X), axis=1) / X.shape[1]
    return strength_vector

#@profile
def compute_strength_matrix_(X):
    notmask = (~np.isnan(X)).astype(np.float32)
    strength_matrix = np.dot(notmask, notmask.T)
    strength_matrix /= X.shape[1]
    return strength_matrix


#@profile
def compute_strength_matrix(X):
    strength_matrix = direct_dot(~np.isnan(X))
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

def demean(S, w):
    w_sum = np.matmul(w, S)
    w_rowsum = w_sum.reshape((1, S.shape[0]))
    w_colsum = w_sum.reshape((S.shape[0], 1))
    w_totalsum = np.dot(w_sum, w)
    S -= (w_rowsum + w_colsum) -  w_totalsum
    return S


def run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename, robust=False):
    start_time = time.time()
    X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
    S, strength_vec, strength_mat = cov(X_incomplete)
    weights_normalized = weights / weights.sum()
    S = demean(S, weights_normalized)
    logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
    W = np.diag(weights)
    if robust: 
        logging.info("Starting matrix completion. This will take a few minutes...")
        start_time = time.time()
        percent_inds_val = 20 # Percent of unmasked individuals to be masked for cross-validation 
        X_incomplete, masked_inds_val = create_validation_mask(X_incomplete.data, percent_inds_val) # masked_inds_val is the list of indices of the individuals masked for validation
        X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
        S_prime, w_vec_prime, w_mat_prime = cov(X_incomplete)
        del X_incomplete
        gc.collect()
        S_prime = demean(S_prime, weights_normalized)
        S, lam = matrix_completion(S, strength_mat, S_prime, w_mat_prime, lams=None, method="NN", 
                cv_inds=masked_inds_val)
        logging.info(f"Covariance Matrix --- %{time.time() - start_time:.2}s seconds ---")
    WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
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


def NN_matrix_completion(cov, w, lam, cov0, verbose=False):
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
    try: 
        X.value = cov0
    except ValueError:
        X.value = None
        logging.info("cov0 ignored")
    obj = cvx.Minimize(lam * cvx.norm(X, "nuc") +
                       cvx.sum_squares(cvx.multiply(np.sqrt(w), cov - X)))
    prob = cvx.Problem(obj, [])
    sol = prob.solve(warm_start=True, solver=cvx.SCS, verbose=verbose)
    return X.value

def cov_to_dist(cov_matrix):
    svd = TruncatedSVD(2, algorithm="arpack")
    X_projected = svd.fit_transform(cov_matrix)
    coordinates_array = np.array(X_projected[:,:2])
    dist_array = pdist(coordinates_array)
    return squareform(dist_array)


def matrix_completion(cov, w, cv_cov, cv_w, cv_inds=None, lams=None, method="NN",
        verbose=False, ncpus=None, parallel=False):
    """ 
    objective: lam ||X||_* + ||w^1/2 (cov - X)||_2^2  with X = PSD 

    Parameters
    ----------
    cov : n x n 
        Input cov matrix to be denoised

    w : n x n
        Weight matrix for confidence in elements of cov 

    verbose : bool, default = False
        verbosity level of cvxpy

    Returns:
    --------
    X : n x n
        Completed matrix
    """
    def dispatcher(method, **kwargs):
        if method == "NN":
            return NN_matrix_completion(**kwargs)

    w *= 1/np.max(w)
    if lams is None:
        lams = np.array([.05,.01, .005, .001, .0001, .00005])
        lams /= np.mean(w)
        lams *= w.shape[0]
    dist = cov_to_dist(cov)
    if cv_inds is not None:
        dist = dist[cv_inds, :][:, cv_inds]
    u, s, v = svds(cov, k=10)
    warm_start = u @ np.diag(s) @ u.T + np.eye(cov.shape[0]) * s[0]/100 
    performance, score  = [], 0
    for i, lam in enumerate(lams): # not parallel
        if lam < 1e-4: 
            continue 
        val = dispatcher(method, **{"cov": cv_cov, "cov0":warm_start, "w": cv_w, "lam":lam})
        if cv_inds is not None:
        # Using a known subset to cv
            cv_dist = cov_to_dist(val)[cv_inds, :][:, cv_inds]
            tmp = mantel(cv_dist, dist)[0]
        if tmp > score:
            l = lam
            score = tmp
    recon = dispatcher(method, **{"cov": cv_cov, "cov0":warm_start, "w": cv_w, "lam":l})
    return recon, l

def scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels):
    plot_df = pd.DataFrame()
    plot_df['x'] = X_projected[:,0]
    plot_df['y'] = X_projected[:,1]
    plot_df['Label'] = labels
    plot_df['ID'] = ind_IDs
    scatter = px.scatter(plot_df, x='x', y='y', color='Label', hover_name='ID', color_discrete_sequence=px.colors.qualitative.Alphabet)
    plotly.offline.plot(scatter, filename = scatterplot_filename, auto_open=False)
    plot_df.to_csv(output_filename, columns=['ID', 'x', 'y'], sep='\t', index=False)
    
def run_method(root_dir, beagle_or_vcf, beagle_vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_fbk_tsv_filename, fb_or_msp, num_ancestries, ancestry, prob_thresh, average_parents, groups_to_remove, min_percent_snps, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masks, load_masks, masks_filename, save_cov_matrix, cov_matrix_filename, robust):
    if not is_masked:
        num_ancestries = 1
        ancestry = '1'
    if load_masks:
        masks, rs_ID_list, ind_ID_list, labels, weights = load_mask_file(masks_filename)
    else:
        num_arrays = 1
        masks, rs_ID_list, ind_ID_list = array_process(root_dir, beagle_vcf_filename, vit_fbk_tsv_filename, beagle_or_vcf, vit_or_fbk_or_tsv, fb_or_msp, num_arrays, num_ancestries, average_parents, prob_thresh, is_masked)
        masks, ind_ID_list, labels, weights = process_labels_weights(labels_filename, masks, rs_ID_list, ind_ID_list, average_parents, num_arrays, ancestry, min_percent_snps, groups_to_remove, is_weighted, save_masks, masks_filename)
    X_incomplete, rs_IDs, ind_IDs = process_masks(masks, rs_ID_list, ind_ID_list, ancestry, is_masked)
    WSW, S, W = run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename, robust)
    X_projected, pc1_percentvar, pc2_percentvar = project_weighted_matrix(WSW, S, W)
    scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels)
    logging.info("Percent variance explained by the 1st principal component: %s", pc1_percentvar)
    logging.info("Percent variance explained by the 2nd principal component: %s", pc2_percentvar)


def get_args(params_file):
    all_args = set(['ROOT_DIR', 'BEAGLE_OR_VCF', 'BEAGLE_VCF_FILE', 'IS_MASKED',
                    'VIT_OR_FBK_OR_TSV', 'VIT_FBK_TSV_FILE', 'FB_OR_MSP',
                    'NUM_ANCESTRIES', 'ANCESTRY', 'PROB_THRESH', 'AVERAGE_PARENTS',
                    'GROUPS_TO_REMOVE', 'MIN_PERCENT_SNPS', 'IS_WEIGHTED', 'LABELS_FILE',
                    'OUTPUT_FILE', 'SCATTERPLOT_FILE', 'SAVE_MASKS', 'LOAD_MASKS', 'MASKS_FILE',
                    'SAVE_COVARIANCE_MATRIX', 'COVARIANCE_MATRIX_FILE', 'ROBUST'])
    int_args = ['BEAGLE_OR_VCF', 'VIT_OR_FBK_OR_TSV', 'FB_OR_MSP', 'NUM_ANCESTRIES']
    bool_args = ['IS_MASKED', 'AVERAGE_PARENTS', 'IS_WEIGHTED', 'SAVE_MASKS', 
                 'LOAD_MASKS', 'SAVE_COVARIANCE_MATRIX', 'ROBUST']
    dict_args = ['GROUPS_TO_REMOVE']
    float_args = ['MIN_PERCENT_SNPS']
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
        args[val] = bool(strtobool(args[val]))
    for val in dict_args:
        args[val] = literal_eval(args[val])
    for val in float_args:
        args[val] = float(args[val])
    logging.debug(f"parameters used are: {args}")
    return args


def run(params_filename):
    args = get_args(params_filename)
    kwargs = {key.lower().replace("covariance", "cov").replace("file", "filename"): 
            args[key] for key in args.keys()}
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
