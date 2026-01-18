"""
Unsupervised Learning Pipeline: Clustering & Structure Discovery

This module performs PCA, K-Means Clustering, and Hierarchical Clustering
to identify latent patterns in the liver patient data.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# ========================================================================
# CONFIGURATION
# ========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ========================================================================
# STEP 1: LOAD PROCESSED DATA
# ========================================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA FOR UNSUPERVISED LEARNING")
print("="*70)

try:
    # We use the training data because it has been resampled and scaled
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_test_processed.csv"))
    print(" Scaled training data loaded successfully.")
    print(f"  Shape: {X_train.shape}")
except FileNotFoundError:
    print(" ERROR: Processed data not found.")
    print("   Please run '_data_preparation.py' first!")
    exit()

# ========================================================================
# STEP 2: PCA FOR VISUALIZATION (DIMENSIONALITY REDUCTION)
# ========================================================================
print("\n" + "="*70)
print("STEP 2: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*70)

# Reduce 10 dimensions down to 2 so we can plot "Patient Similarity"
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Calculate explained variance (How much info did we lose?)
explained_var = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: PC1={explained_var[0]:.2f}, PC2={explained_var[1]:.2f}")
print(f"Total Information Retained: {sum(explained_var)*100:.1f}%")

# Create a DataFrame for easy plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# ========================================================================
# STEP 3: K-MEANS CLUSTERING
# ========================================================================
print("\n" + "="*70)
print("STEP 3: K-MEANS CLUSTERING")
print("="*70)

# 3A. The Elbow Method (Finding the optimal K)
print("--- Calculating optimal K (Elbow Method) ---")
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Variance)')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "05_kmeans_elbow.png"))
plt.close()
print("Saved: 05_kmeans_elbow.png")

# 3B. Run K-Means with K=3 (Hypothesis: Healthy, Mild, Severe)
k_optimal = 3
print(f"\n--- Running K-Means with K={k_optimal} ---")

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train)

# Add cluster labels to our PCA dataframe
pca_df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=80)
plt.title(f'Patient Clusters (K={k_optimal}) Visualized on PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "06_pca_clusters.png"))
plt.close()
print("Saved: 06_pca_clusters.png")

# ========================================================================
# STEP 4: HIERARCHICAL CLUSTERING (DENDROGRAM)
# ========================================================================
print("\n" + "="*70)
print("STEP 4: HIERARCHICAL CLUSTERING")
print("="*70)

# We take a sample of 50 patients because a dendrogram with 400+ lines is unreadable
sample_idx = np.random.choice(len(X_train), 50, replace=False)
X_sample = X_train.iloc[sample_idx]

#  logic here
plt.figure(figsize=(12, 6))
# 'ward' linkage minimizes variance within clusters
linked = linkage(X_sample, method='ward')

dendrogram(linked, 
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)

plt.title('Hierarchical Clustering Dendrogram (50 Random Patients)')
plt.xlabel('Patient Index')
plt.ylabel('Euclidean Distance')
plt.savefig(os.path.join(OUTPUT_DIR, "07_dendrogram.png"))
plt.close()
print("Saved: 07_dendrogram.png")

# ========================================================================
# STEP 5: CLINICAL INTERPRETATION OF CLUSTERS
# ========================================================================
print("\n" + "="*70)
print("STEP 5: INTERPRETING THE CLUSTERS")
print("="*70)

# We attach the cluster IDs back to the feature data to see the "average patient" in each group
X_analysis = X_train.copy()
X_analysis['Cluster'] = clusters

# Calculate the mean value of each feature for each cluster
cluster_profile = X_analysis.groupby('Cluster').mean()

print("\n--- Cluster Profiles (Mean Scaled Values) ---")
print("Note: Values are scaled (Z-scores). 0 = Average, >0 = High, <0 = Low")
print(cluster_profile)

# Heatmap of cluster centers for easy reading
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile.T, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
plt.title('Clinical Profile of Each Cluster (Scaled Values)')
plt.xlabel('Cluster ID')
plt.ylabel('Feature')
plt.savefig(os.path.join(OUTPUT_DIR, "08_cluster_profile_heatmap.png"), bbox_inches='tight')
plt.close()
print("Saved: 08_cluster_profile_heatmap.png")

print("\n" + "="*70)
print("UNSUPERVISED LEARNING COMPLETE.")
print("="*70)