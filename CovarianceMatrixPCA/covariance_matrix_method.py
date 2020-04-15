import sys
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
import gc
import matplotlib.pyplot as plt
import plotly_express as px
import plotly
from distutils.util import strtobool
from file_processing import get_masked_matrix, process_labels_weights

def cov(x):
    ddof = 1
    x = np.ma.array(x, ndmin=2, copy=True, dtype=np.float32)
    xmask = np.ma.getmaskarray(x)
    rowvar = 1
    axis = 1 - rowvar
    tup = (slice(None), None)
    xnotmask = np.logical_not(xmask).astype(np.float32)
    fact = np.dot(xnotmask, xnotmask.T) * 1. - ddof
    del(xnotmask)
    gc.collect()
    result = (np.ma.dot(x, x.T, strict=False) / fact).squeeze()
    return result

def run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename):
    start_time = time.time()
    X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
    S = cov(X_incomplete).data
    weights_normalized = weights / weights.sum()
    weighted_sum = np.matmul(weights_normalized, S)
    weighted_rowsum = weighted_sum.reshape((1, S.shape[0]))
    weighted_colsum = weighted_sum.reshape((S.shape[0], 1))
    weighted_totalsum = np.dot(weighted_sum, weights_normalized)
    S = S - weighted_rowsum - weighted_colsum + weighted_totalsum
    print("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
    W = np.diag(weights)
    WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
    if save_cov_matrix:
        np.save(cov_matrix_filename, S.data)
    return WSW, S, W

def project_weighted_matrix(WSW, S, W):
    svd = TruncatedSVD(2, algorithm="arpack")
    svd.fit(WSW)
    X_projected = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
    total_var = np.trace(S)
    pc1_percentvar = 100 * svd.singular_values_[0] / total_var
    pc2_percentvar = 100 * svd.singular_values_[1] / total_var
    return X_projected, pc1_percentvar, pc2_percentvar

def scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels):
    plot_df = pd.DataFrame()
    plot_df['x'] = X_projected[:,0]
    plot_df['y'] = X_projected[:,1]
    plot_df['Label'] = labels
    plot_df['ID'] = ind_IDs
    scatter = px.scatter(plot_df, x='x', y='y', color='Label', hover_name='ID', color_discrete_sequence=px.colors.qualitative.Alphabet)
    plotly.offline.plot(scatter, filename = scatterplot_filename, auto_open=False)
    plot_df.to_csv(output_filename, columns=['ID', 'x', 'y'], sep='\t', index=False)
    
def run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_cov_matrix, cov_matrix_filename):
    X_incomplete, ind_IDs, rs_IDs = get_masked_matrix(beagle_filename, vcf_filename, beagle_or_vcf, is_masked, vit_filename, fbk_filename, tsv_filename, vit_or_fbk_or_tsv, fb_or_msp, num_ancestries, ancestry, average_parents, prob_thresh)
    X_incomplete, ind_IDs, labels, weights = process_labels_weights(labels_filename, X_incomplete, ind_IDs, average_parents, is_weighted, save_masked_matrix, masked_matrix_filename)
    WSW, S, W = run_cov_matrix(X_incomplete, weights, save_cov_matrix, cov_matrix_filename)
    X_projected, pc1_percentvar, pc2_percentvar = project_weighted_matrix(WSW, S, W)
    scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels)
    print("Percent variance explained by the 1st principal component: ", pc1_percentvar)
    print("Percent variance explained by the 2nd principal component: ", pc2_percentvar)

def run(params_filename):
    file = open(params_filename)
    params = {}
    for line in file:
        line = line.strip()
        if not line.startswith('#'):
            key_value = line.split('=')
            if len(key_value) == 2:
                params[key_value[0].strip()] = key_value[1].strip()
    file.close()
    beagle_or_vcf = int(params['BEAGLE_OR_VCF'])
    beagle_filename = str(params['BEAGLE_FILE'])
    vcf_filename = str(params['VCF_FILE'])
    is_masked = bool(strtobool(params['IS_MASKED']))
    vit_or_fbk_or_tsv = int(params['VIT_OR_FBK_OR_TSV'])
    vit_filename = str(params['VIT_FILE'])
    fbk_filename = str(params['FBK_FILE'])
    fb_or_msp = int(params['FB_OR_MSP'])
    tsv_filename = str(params['TSV_FILE'])
    num_ancestries = int(params['NUM_ANCESTRIES'])
    ancestry = int(params['ANCESTRY'])
    prob_thresh = float(params['PROB_THRESH'])
    average_parents = bool(strtobool(params['AVERAGE_PARENTS']))
    is_weighted = bool(strtobool(params['IS_WEIGHTED']))
    labels_filename = str(params['LABELS_FILE'])
    output_filename = str(params['OUTPUT_FILE'])
    scatterplot_filename = str(params['SCATTERPLOT_FILE'])
    save_masked_matrix = bool(strtobool(params['SAVE_MASKED_MATRIX']))
    masked_matrix_filename = str(params['MASKED_MATRIX_FILE'])
    save_cov_matrix = bool(strtobool(params['SAVE_COVARIANCE_MATRIX']))
    cov_matrix_filename = str(params['COVARIANCE_MATRIX_FILE'])
    run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_cov_matrix, cov_matrix_filename)
    
def main():
    params_filename = sys.argv[1]
    start_time = time.time()
    run(params_filename)
    print("Total time --- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()