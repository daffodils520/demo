# -----------------------------------------------------------------------------
# Import required libraries
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings

warnings.filterwarnings("ignore")

# Fix Chinese font issues on Windows (can remain even for English plots)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. Load optimized fragment matrix
# -----------------------------------------------------------------------------
data_file = "./demo/demo_optimized_fragment_matrix.csv"
matrix_df = pd.read_csv(data_file, index_col=0)
data_matrix = matrix_df.values

# Set random seed
np.random.seed(42)

# -----------------------------------------------------------------------------
# 2. Set number of components for NMF
# -----------------------------------------------------------------------------
n_components = 5

# -----------------------------------------------------------------------------
# 3. Initialize W_init and H_init (random initialization)
# -----------------------------------------------------------------------------
W_init = np.abs(np.random.rand(data_matrix.shape[0], n_components))
H_init = np.abs(np.random.rand(n_components, data_matrix.shape[1]))

# -----------------------------------------------------------------------------
# 4. Perform NMF (custom initialization)
# -----------------------------------------------------------------------------
nmf_model = NMF(
    n_components=n_components,
    init='custom',
    random_state=42,
    max_iter=1000
)

W_matrix = nmf_model.fit_transform(data_matrix, W=W_init, H=H_init)
H_matrix = nmf_model.components_

# -----------------------------------------------------------------------------
# 5. Save W & H matrices
# -----------------------------------------------------------------------------
save_dir = "./demo/NMF_results"
os.makedirs(save_dir, exist_ok=True)

component_names = [f'Component_{i+1}' for i in range(n_components)]
W_df = pd.DataFrame(W_matrix, index=matrix_df.index, columns=component_names)
H_df = pd.DataFrame(H_matrix, index=component_names, columns=matrix_df.columns)

W_output_file = os.path.join(save_dir, "NMF_W_matrix_n5.csv")
H_output_file = os.path.join(save_dir, "NMF_H_matrix_n5.csv")
W_df.to_csv(W_output_file)
H_df.to_csv(H_output_file)

print(f"NMF completed! W matrix saved to: {W_output_file}")
print(f"H matrix saved to: {H_output_file}")

# -----------------------------------------------------------------------------
# 6. Extract top 15 m/z features per component
# -----------------------------------------------------------------------------
top_features_list = []
for component in component_names:
    top_mz = H_df.loc[component].nlargest(15).index.tolist()
    for mz in top_mz:
        top_features_list.append([component, mz])

top_features_file = os.path.join(save_dir, "top_features_per_component_n5.csv")
top_features_df = pd.DataFrame(top_features_list, columns=["Component", "Top_m/z"])
top_features_df.to_csv(top_features_file, index=False)

print(f"Top m/z features saved to: {top_features_file}")

# -----------------------------------------------------------------------------
# 7. Hierarchical clustering on samples (W matrix)
# -----------------------------------------------------------------------------
linkage_matrix = linkage(W_df, method='ward', metric='euclidean')
ordered_indices = leaves_list(linkage_matrix)
W_df_sorted = W_df.iloc[ordered_indices, :]

# -----------------------------------------------------------------------------
# 8. Remove samples with extremely low contribution
# -----------------------------------------------------------------------------
low_contribution_samples = W_df_sorted.sum(axis=1) < 0.01
print(f"Number of low-contribution samples removed: {low_contribution_samples.sum()}")

W_df_sorted_filtered = W_df_sorted[~low_contribution_samples]

# -----------------------------------------------------------------------------
# 9. Color mapping for components (soft color palette)
# -----------------------------------------------------------------------------
custom_palette = [
    "#E46A6A", "#64B5F6", "#81C784", "#FFD54F",
    "#C37BCF", "#4DB6AC", "#F27AA2", "#FCBB74",
    "#A1887F", "#90A4AE", "#7986CB", "#DCE775"
]
color_dict = {component: color for component, color in zip(component_names, custom_palette)}

# -----------------------------------------------------------------------------
# 10. Plot H matrix (m/z vs component weights)
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 6), dpi=300)
for component in component_names:
    plt.plot(
        H_df.columns,
        H_df.loc[component, :],
        label=component,
        linewidth=1.2,
        alpha=0.8,
        color=color_dict[component]
    )

plt.xlabel("m/z (original order)", fontsize=12)
plt.ylabel("Component Weight", fontsize=12)
plt.title("NMF Component vs m/z Relationship", fontsize=14)

xticks_interval = max(len(H_df.columns) // 15, 1)
plt.xticks(
    range(0, len(H_df.columns), xticks_interval),
    H_df.columns[::xticks_interval],
    rotation=45,
    fontsize=10
)

plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

h_matrix_plot_path = os.path.join(save_dir, "NMF_H_Matrix_Visualization_n5.png")
plt.savefig(h_matrix_plot_path, dpi=300, bbox_inches='tight')
print(f"H matrix visualization saved to: {h_matrix_plot_path}")

plt.show()

# -----------------------------------------------------------------------------
# 11. Plot W matrix (sample contributions)
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 6), dpi=300)

W_df_normalized = W_df_sorted_filtered.div(
    W_df_sorted_filtered.sum(axis=1),
    axis=0
)

ax1 = W_df_normalized.plot(
    kind="bar",
    stacked=True,
    width=0.8,
    figsize=(14, 6),
    color=[color_dict[comp] for comp in W_df.columns]
)

ax1.set_xlabel("Samples (clustered)", fontsize=12)
ax1.set_ylabel("Component Contribution Proportion", fontsize=12)
ax1.set_title("Sample vs Component Relationship (Clustered)", fontsize=14)

xticks_interval = max(len(W_df_sorted_filtered) // 20, 1)
ax1.set_xticks(range(0, len(W_df_sorted_filtered), xticks_interval))
ax1.set_xticklabels(
    W_df_sorted_filtered.index[::xticks_interval],
    rotation=45,
    fontsize=10
)

ax1.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

w_matrix_plot_path = os.path.join(save_dir, "NMF_W_Matrix_Visualization_n5.png")
plt.savefig(w_matrix_plot_path, dpi=300, bbox_inches='tight')
print(f"W matrix visualization saved to: {w_matrix_plot_path}")

plt.show()
