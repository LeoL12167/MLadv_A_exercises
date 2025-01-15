import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorcet as cc



#read similarity matrix into pandas dataframe and then convert to numpy array

similarity_matrix_ext = pd.read_csv("MEP_similarity_matrix.csv", header=None).to_numpy()

#read MEP info into pandas dataframe and then convert to numpy array
info_array = pd.read_csv("MEP_info.csv", index_col=0).to_numpy()

epgs = info_array[:,1] 
country_list = info_array[:,0]

#make sure to exclude headers 
similarity_matrix = similarity_matrix_ext[1:,1:]

#----------------------------------------------------------------------
# 8. Define the distance matrix as sqrt(1-similarity)
#----------------------------------------------------------------------
def similarity_to_distance(similarity_matrix):
    """
    Convert a similarity matrix into a distance matrix.
    The distance is defined as sqrt(1 - similarity).
    """
    # Compute the distance matrix
    safe_diff = 1 - np.clip(similarity_matrix, 0, 1)
    distance_matrix = np.sqrt(safe_diff)

    print("Min similarity:", np.nanmin(similarity_matrix))
    print("Max similarity:", np.nanmax(similarity_matrix))

    return distance_matrix

#----------------------------------------------------------------------
# 9. Compute classical MDS
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

#----------------------------------------------------------------------
# 10.Compute Distance MAtrix and apply MDS to the distance matrix
#----------------------------------------------------------------------

dist_matrix = similarity_to_distance(similarity_matrix)
coords_2d = classical_mds(dist_matrix, dim=2)


#----------------------------------------------------------------------
# 11. Plot the 2D coordinates
#----------------------------------------------------------------------
if __name__ == "__main__":
    plt.figure(figsize=(8,6))


    unique_epgs = list(set(epgs))
    colors = plt.cm.get_cmap('tab10', len(unique_epgs))  # or another colormap

    for epg in unique_epgs:
        idx = [i for i in range(len(epgs)) if epgs[i] == epg]
        plt.scatter(coords_2d[idx, 0], coords_2d[idx, 1], 
                    label=epg, alpha=0.7, c=[colors(unique_epgs.index(epg))])

    plt.title("MDS Embedding Colored by EPG")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1, 1), 
            loc='upper left', 
            borderaxespad=0.)
    plt.savefig("MDS_EPG.png", bbox_inches="tight")
    plt.show()

    #----------------------------------------------------------------------
    plt.figure(figsize=(8,6))

    unique_countries = sorted(set(country_list))
    num_countries = len(unique_countries)  # 28 in your case

    cmap = cc.glasbey[:num_countries]

    for i, country in enumerate(unique_countries):
        idx = [j for j, c in enumerate(country_list) if c == country]
        plt.scatter(coords_2d[idx, 0],
                    coords_2d[idx, 1],
                    label=country,
                    color=[cmap[i]],
                    s=60,
                    edgecolor='k',
                    linewidth=0.5)

    plt.title("MDS Embedding Colored by Country")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1, 1), 
            loc='upper left', 
            borderaxespad=0.)
    plt.savefig("MDS_Country.png", bbox_inches="tight")
    plt.show()