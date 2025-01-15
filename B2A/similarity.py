# Legend:
# 0 - not a MEP
# 1 - for
# 2 - against
# 3 - abstention
# 4 - absent
# 5 - didn't vote

import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# The CSV has columns: 
#   ['MEP_ID', 'Country', 'EPG', 'Vote_1', 'Vote_2', ..., 'Vote_M']
# where each 'Vote_i' is an integer from 0 to 5 as per legend.

df = pd.read_excel("B2A\EP9_RCVs_2022_06_22.xlsx", sheet_name=0)

keys = df.keys()
print(keys[:10])


# ----------------------------------------------------------------------
# 1. Define how to convert each integer code into a numeric format
#    suitable for similarity calculations
# ----------------------------------------------------------------------
def convert_vote_to_numeric(v):
    """
    Convert the legend-based vote code into a numeric value:
      - 1 (for)       -> +1
      - 2 (against)   -> -1
      - 3 (abstention)->  0
      - 4 (absent)    ->  np.nan (skip in similarity)
      - 5 (didn't vote)-> np.nan (skip in similarity)
      - 0 (not a MEP) -> np.nan or exclude row entirely
    """
    if v == 1:       # for
        return  1
    elif v == 2:     # against
        return -1
    elif v == 3:     # abstention
        return  0
    elif v in [4, 5, 0]:  
        # absent / didn't vote / not a MEP
        # We'll treat these as missing for the similarity measure
        return np.nan
    else:
        return np.nan

# ----------------------------------------------------------------------
# 2. Convert all 'Vote_X' columns using the above function
# ----------------------------------------------------------------------
all_cols = df.columns
vote_cols = all_cols[10:]  # from the 11th column onward
for col in vote_cols:
    df[col] = df[col].apply(convert_vote_to_numeric)

# ----------------------------------------------------------------------
# 3. Subset data if needed (e.g., remove rows where MEP_ID is invalid).
#    If "0 - not a MEP" was in the data, you might exclude those rows:
# ----------------------------------------------------------------------
#df = df.dropna(subset=["MEP_ID"])  # or any logic specific to your dataset

# ----------------------------------------------------------------------
# 4. Create vote matrix (N x M)
# ----------------------------------------------------------------------

key = keys[0]
mep_ids = df[key].unique()

N = len(mep_ids)
M = len(vote_cols)

vote_matrix = np.zeros((N, M))
for i, mep_id in enumerate(mep_ids):
    # If each MEP has a single row, grab that row's votes
    row_data = df.loc[df[key] == mep_id, vote_cols].iloc[0].values
    vote_matrix[i] = row_data

# ----------------------------------------------------------------------
# 5. Define similarity functions
# ----------------------------------------------------------------------

# Agreement rate (fraction of votes where both MEPs match, ignoring NaNs)
def agreement_rate(vec1, vec2):
    mask = ~np.isnan(vec1) & ~np.isnan(vec2)
    if mask.sum() == 0:
        return 0
    same_votes = np.sum(vec1[mask] == vec2[mask])
    return same_votes / mask.sum()

# Cosine similarity (using only overlapping non-NaN votes)
def cosine_similarity(vec1, vec2):
    mask = ~np.isnan(vec1) & ~np.isnan(vec2)
    if mask.sum() == 0:
        return 0
    v1 = vec1[mask]
    v2 = vec2[mask]
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

# Choose which similarity measure to use
SIMILARITY_METHOD = "cosine"  # or "agreement"

def compute_similarity(vec1, vec2, method="cosine"):
    if method == "cosine":
        return cosine_similarity(vec1, vec2)
    elif method == "agreement":
        return agreement_rate(vec1, vec2)
    else:
        raise ValueError("Unknown similarity method: %s" % method)

# ----------------------------------------------------------------------
# 6. Compute pairwise similarity matrix
# ----------------------------------------------------------------------
similarity_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        sim = compute_similarity(vote_matrix[i], vote_matrix[j], 
                                 method=SIMILARITY_METHOD)
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim
        print(f"Similarity between {mep_ids[i]} and {mep_ids[j]}: {sim}")
    print(f"Line {i} out of  {N} done")

# Optionally set the diagonal to 1 (similarity with oneself)
np.fill_diagonal(similarity_matrix, 1.0)

# ----------------------------------------------------------------------
# 7. Save or analyze similarity matrix
# ----------------------------------------------------------------------
sim_df = pd.DataFrame(similarity_matrix, index=mep_ids, columns=mep_ids)
sim_df.to_csv(f"MEP_similarity_matrix_{SIMILARITY_METHOD}.csv")

print("Similarity matrix computed using:", SIMILARITY_METHOD)
print(sim_df.head())
