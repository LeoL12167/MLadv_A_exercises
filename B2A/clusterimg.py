from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

from MDS import similarity_matrix, dist_matrix, coords_2d, epgs
# ----------------------------------------------------------------------	
# 12. Hierarchical clustering
# ----------------------------------------------------------------------

tri_upper = np.triu_indices(dist_matrix.shape[0], k=1)
condensed_dist = dist_matrix[tri_upper]

groups = epgs
N = len(groups)

# Assign a color to each group
unique_groups = set(groups)
group_colors = {group: plt.cm.tab10(i) for i, group in enumerate(unique_groups)}

# Perform hierarchical clustering (e.g., ward linkage)
leaf_colors = [group_colors[group] for group in groups]

# Convert the distance matrix to condensed form (required for `linkage`)


# Perform hierarchical clustering
Z = linkage(condensed_dist, method='ward')

# Function to determine branch color
def get_dendrogram_leaves(linkage_matrix):
    tree, nodes = to_tree(linkage_matrix, rd=True)
    node_to_leaves = defaultdict(list)

    def collect_leaves(node):
        if node.is_leaf():
            node_to_leaves[node.id].append(node.id)
        else:
            for child in node.pre_order(lambda x: x.id):
                node_to_leaves[node.id].append(child)

    collect_leaves(tree)
    return node_to_leaves

# Get the mapping of nodes to their descendant leaves
node_to_leaves = get_dendrogram_leaves(Z)

# Define the branch color function
def branch_color_func(node_id):
    """
    Determine the branch color:
    - If all leaves in the cluster belong to the same group, use that group's color.
    - If the cluster contains mixed groups, use a neutral color (e.g., grey).
    """
        # Get all leaves under this branch
    leaves = node_to_leaves[node_id]
    leaf_groups = {groups[leaf] for leaf in leaves}  # Set of unique groups
    leaf_groups = set(leaf_groups)
    
    if len(leaf_groups) == 1:
        # All leaves in this branch belong to the same group
        group = leaf_groups.pop()
        return group_colors[group]
    else:
        # Mixed groups, use neutral color
        return 'grey'

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=10,
    labels=groups,
    link_color_func=branch_color_func  # Use custom branch color logic
)
plt.title("Dendrogram with Branch Coloring by Group")
plt.xlabel("MEP Grouops")
plt.ylabel("Distance")
plt.savefig("dendrogram.png", bbox_inches="tight")
plt.show()

# Decide how many clusters to cut
num_clusters = 2
clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

# 'clusters' is an array of length N, telling which cluster each MEP belongs to
# Compare with EPG or country:
for cluster_id in range(1, num_clusters + 1):
    # Indices of MEPs that belong to cluster_id
    members = np.where(clusters == cluster_id)[0]
    # The EPGs of those MEPs
    cluster_epgs = [epgs[m] for m in members]
    
    # Count how many times each EPG appears
    epg_count = Counter(cluster_epgs)
    
    # Print out or store results
    print(f"\nCluster {cluster_id} (size={len(members)}) EPG counts:")
    for epg, count in epg_count.items():
        print(f"  {epg}: {count}")