import time
import torch
import math
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import csv

torch.manual_seed(1)
np.random.seed(1)

def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]


def scipy_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scipy's KDTree
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d, nu_i = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k + 1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
    else:
        D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))

    return D



KLdivs = [0]*49
for j in range(1,50):
    
    file_Samples = "C:/Users/amval/PyWork/CausalSamps2/RepeatedDraws/Draw_Post_" + str(j) + ".csv"
    Data_Norm = pd.read_csv(file_Samples)

    file_Samples = "C:/Users/amval/PyWork/CausalSamps2/RepeatedDraws/Truth_" + str(j) + ".csv"
    Data_Unif = pd.read_csv(file_Samples)

    KLdivs[j-1] = scipy_estimator(Data_Norm, Data_Unif, k=5)

print(np.average(KLdivs))
