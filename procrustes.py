import numpy as np
from scipy.linalg import svd


def procrustes(X, Y):
    '''Solve for linear mapping using procrustes solution.'''
    
    M = X.T@Y
    U, S, V_T = svd(M, full_matrices=True )
    W = U.dot(V_T)

    return W




