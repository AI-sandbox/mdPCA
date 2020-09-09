import numpy as np
from scipy.linalg import eigh
from numpy.linalg import norm
from numpy.linalg import svd
import tqdm
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd, svd_flip
import pdb
from scipy.linalg import eigh
from numpy.linalg import norm
from sklearn.impute import SimpleImputer
from time import time
from sklearn.decomposition import TruncatedSVD
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform
import cvxpy as cvx
import pathos.multiprocessing as mp

def cov_to_dist(cov_matrix):
    svd = TruncatedSVD(2, algorithm="arpack")
    X_projected = svd.fit_transform(cov_matrix)
    coordinates_array = np.array(X_projected[:,:2])
    dist_array = pdist(coordinates_array)
    dist_matrix = squareform(dist_array)
    return dist_matrix


def weighted_NN(cov, w, lam, cov0=None):
    TOL = 1e-4
    VERBOSE = False
    M = cov.shape[0]
    constraints = []
    X = cvx.Variable(shape=(M, M), PSD=True)
    if cov0 is None:
        u, s, v = svds(cov, k=10)
        cov0 = u@np.diag(s)@u.T # warm_start 
    X.value = cov0
    obj = cvx.Minimize(lam * cvx.norm(X, "nuc") +
        cvx.sum_squares(cvx.multiply(np.sqrt(w), cov - X)))
    prob = cvx.Problem(obj, constraints)
    sol = prob.solve(warm_start=True, solver=cvx.SCS, verbose=VERBOSE)
    #sol = prob.solve(solver=cvx.SCS)
    return X.value


def prox_solver(mat, mask, rho=None, weights=None):
  """docstring"""
  # Figure out rho
  d =  mask.shape[0] # square matrix 
  grid_n = 100 # roughly
  grid_n = max(grid_n, 10)
  if isinstance(weights, (list, np.ndarray)):
    w = np.diag(weights)
    mat = w @ mat @ w
  cv_mat = mat.copy()
  p_cv = .05
  tot_known = np.sum(~mask)
  n_cv = int(tot_known * p_cv)
  mask_cv = mask.copy()# np.zeros_like(mask, dtype=bool)
  rows =  np.random.choice(d, size=n_cv)
  cols = np.random.choice(d, size=n_cv)
  mask_cv[rows, cols] = True 
  mask_cv *= mask_cv
  n_cv -= (np.sum(mask_cv) - np.sum(mask))//2 # Don't care about \pm 1    
  while n_cv > 0:
    indx =  np.random.choice(d, size=2) 
    if not mask_cv[indx[0], indx[1]]:
      mask_cv[indx[0], indx[1]] = True 
      mask_cv[indx[1], indx[0]] = True 
      n_cv -= 1
  mask_cv = np.logical_and(mask_cv, ~mask)
  cv_mat[mask_cv] = 0
  U, S, V = svd(cv_mat)
  grid_q = list(filter(lambda x: x > 10/d, list(np.array(range(grid_n//2, grid_n))/grid_n)))
  grid_q = [1/d] + grid_q + list(1-np.array(range(1, grid_n//2))/d)
  grid_q =  list(1-np.array(range(1, grid_n//2))/d)
  rho_grid = np.quantile(S, grid_q)
  cv_vals = []
  for rho in tqdm.tqdm(rho_grid):
    cv_val = compare_prox(U, S, mask_cv, mat, rho)
    cv_vals.append(cv_val)
  idx = np.argmin(cv_vals)
  opt_rho =  rho_grid[idx]
  print(idx)
  x, rnk = prox(U, S, opt_rho)
  if weights is not None:
    invw = np.diag(1/weights)
    x  = invw @ x @ invw
  print(f"Chosen rank is {rnk}")
  return x

  
def compare_prox(U, S, mask, unmasked, rho):
  x = prox(U, S, rho)[0]  
  return (norm(unmasked[mask] - x[mask]))

def prox(U, S, rho):
  rho -= 1e-5
  diags = np.maximum(S-rho, 0)
  U = U[:, diags>0]
  diags = diags[diags>0]
  diags[diags>0] += rho
  rnk = len(diags)
  diags = np.diag(diags)
  return U.dot(diags).dot(U.T), rnk


def k_svd(m, k):
  U, S, V = svds(m, k=k)
  S = S[::-1]
  U, V =  svd_flip(U[:,::-1], V[::-1])
  return U, S, V

def svt(mat, mask, rho=None, weights=None):
  TOL = 1e-3
  tau = 5*np.linalg.norm(mat)# 5*np.prod(mat.shape)
  if weights is None:
    weights = ~mask
  #delta = 1.2 * np.prod(mat.shape) / np.sum(weights)
  delta =  1.2 *  1/np.mean(weights)
  print(tau, delta)
  r_old = 0
  y = np.zeros_like(mat)
  ell = 10
  for iter in range(200):
    if iter == 0: 
      x = np.zeros_like(mat)
    else: 
      sk = r_old + 1
      while True:
        U, S, V = k_svd(y, sk)  # make this more efficient
        if S[-1] < tau:
          break
        sk += ell
      diags = np.maximum(S-tau, 0)
      r_old = np.sum(diags>0)
      U = U[:, :r_old]
      diags = diags[:r_old]
      diags = np.diag(diags)
      x = U.dot(diags).dot(U.T)
    y += delta * weights *(mat - x)
    err = np.linalg.norm(weights * (x-mat)) / np.linalg.norm(weights * mat)
    print(f"Error: {err}, rank: {r_old}")
    if err < TOL: 
      break



#def nestrov(mat):
 

if __name__ == "__main__":
  #np.random.seed(1223)
  #N = 2000
  #s = 10
  #U = np.random.randn(N, s)
  #diag = np.diag(20*np.repeat(1, s))
  #mat = np.linalg.multi_dot([U, diag, U.T])
  #noise = np.random.randn(N, N)
  #noise = 2* noise.dot(noise.T)
  #mat += noise
  #print(f"SNR={np.linalg.norm(mat)/np.linalg.norm(noise)}")
  #mask = (np.random.random(size=(N,N))>.5).astype(bool)
  #mask *= mask.T
  #missing = mat.copy()
  #missing[mask] = 0
  #print(f"frac masked is {np.mean(mask)}") 
  #x = svt(missing, mask)
  #print("done")
  #x = prox_solver(missing, mask)
  #print (norm(mat[mask] - x[mask])/np.sum(mask), norm(mat[mask] - np.mean(missing[~mask]))/np.sum(mask))
  #print (norm(mat[~mask] - x[~mask])/np.sum(~mask))


  data_dir = "POPRES_TestData"
  cov0 = np.load(f"{data_dir}/cov_matrix_POPRES_20_0%masked.npy")

  data_dir = "POPRES_TestData_50"
  rscores50 = []
  oscores50 = []
  cvscores50 = []
  lams = [.1, .01, .001, .0001]
  vals = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98]
  lambs50 = []
  dist0 = cov_to_dist(cov0)
  ncpus = 2
  with mp.ProcessingPool(ncpus) as p:
  #if True: 
    for i, val in enumerate(vals):
       cov = np.load(f"{data_dir}/cov_matrix_POPRES_20_{val}%masked.npy")
       w = np.load(f"{data_dir}/strength_matrix_POPRES_20_{val}%masked.npy")
       cvCov = np.load(f"{data_dir}/cov_matrix_POPRES_20_{vals[i+1]}%masked.npy")
       cvW = np.load(f"{data_dir}/strength_matrix_POPRES_20_{vals[i+1]}%masked.npy")
       dist = cov_to_dist(cov)
       score = 0
       oscores50.append(mantel(dist, dist0)[0])
       print(i)
       u, s, v = svds(cvCov, k=10)
       cov0 = u@np.diag(s)@u.T # warm_start 
       t = time()
       #for lam in lams: 
       #    weighted_NN(cvCov, cvW, lam, cov0)
       cv_vals = p.map(weighted_NN,(cvCov, cvCov, cvCov, cvCov), (cvW, cvW, cvW, cvW), lams, (
           cov0, cov0, cov0, cov0) )
           print(time()-t)
       tmp = mantel(cov_to_dist(val), dist)[0]
       if tmp > score: 
          l = lam
          score = tmp 
       cvscores50.append(score)
       lambs50.append(l)
       val = weighted_NN(cov, w, lam=l)
       rscores50.append(mantel(cov_to_dist(val), dist0)[0]) 
       break
       plt.plot( vals[:i+1], rscores50)
       plt.plot( vals[:i+1], oscores50)
       plt.show()
