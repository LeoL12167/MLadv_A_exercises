import numpy as np
import pandas as pd

#read similarity matrix into pandas dataframe and then convert to numpy array

similarity_matrix_ext = pd.read_csv("MEP_similarity_matrix.csv", header=None).to_numpy()

#make sure to exclude headers 
similarity_matrix = similarity_matrix_ext[1:,1:]

#----------------------------------------------------------------------
# 4. Define the distance matrix as sqrt(1-similarity)
#----------------------------------------------------------------------
def similarity_to_distance(similarity_matrix):
    """
    Convert a similarity matrix into a distance matrix.
    The distance is defined as sqrt(1 - similarity).
    """
    # Compute the distance matrix
    distance_matrix = np.sqrt(1 - similarity_matrix)
    return distance_matrix

#----------------------------------------------------------------------
# 5. Compute classical MDS
#----------------------------------------------------------------------

def classical_mds(dist_matrix, dim=2):
    """
    Classic MDS using eigen-decomposition.
    dist_matrix: NxN matrix of distances
    dim: dimension for the output (usually 2)
    Returns: Nx(dim) array of coordinates.
    """
    N = dist_matrix.shape[0]
    
    # 1. Square the distances
    D_sq = dist_matrix ** 2
    
    # 2. Double-centering
    # H = I - 1/N * 11^T
    I = np.eye(N)
    ones = np.ones((N,1))
    H = I - (1.0/N) * (ones @ ones.T)
    
    # B = -1/2 * H * D_sq * H
    B = -0.5 * H @ D_sq @ H
    
    # 3. Eigen-decomposition of B
    # eigenvals, eigenvecs = np.linalg.eigh(B)
    #  (Use eigh because B should be symmetric)
    eigenvals, eigenvecs = np.linalg.eigh(B)
    
    # Sort eigenvalues (descending)
    idx_sort = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx_sort]
    eigenvecs = eigenvecs[:, idx_sort]
    
    # 4. Take top 'dim' positive eigenvalues
    # If an eigenvalue is negative (numerical issues), treat as 0
    pos_eigs = np.maximum(eigenvals[:dim], 0.0)
    Lambda_half = np.diag(np.sqrt(pos_eigs))
    
    # ...and the corresponding eigenvectors
    V = eigenvecs[:, :dim]
    
    # 5. Coordinates = V * sqrt(Lambda)
    X = V @ Lambda_half
    return X



