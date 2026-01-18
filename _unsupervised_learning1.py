# =========================================================
# UNSUPERVISED LEARNING PIPELINE
# Strategy: Isolation Forest (Cleaning) + UMAP (Manifold) + PSO-KMeans
# Includes: Total Composite Score Calculation
# =========================================================

import numpy as np
import pandas as pd
import os
import sys
import joblib

# Standard ML Imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import f_oneway, kruskal

# PSO Optimization
try:
    import pyswarms as ps
except ImportError:
    print("❌ Error: 'pyswarms' not installed. Run: pip install pyswarms")
    sys.exit()

# UMAP (Critical for High Metrics)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    print("❌ Error: 'umap-learn' not installed. This strategy REQUIRES UMAP.")
    print("   Run: pip install umap-learn")
    sys.exit()

# =========================================================
# CONFIGURATION & PATHS
# =========================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Robust Path Definition
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "X_test_processed.csv")
LABEL_PATH = os.path.join(CURRENT_DIR, "data", "processed", "y_test_processed.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
MODEL_DIR = os.path.join(CURRENT_DIR, "saved_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 1. LOAD & PREPARE DATA
# =========================================================
print(f"Loading data from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print("❌ Error: Processed data not found. Run _data_preparation.py first.")
    sys.exit()

X_full = pd.read_csv(DATA_PATH)
y_true_full = pd.read_csv(LABEL_PATH).values.ravel() if os.path.exists(LABEL_PATH) else None

# Ensure no NaNs
X_clean = X_full.dropna()
if len(X_full) != len(X_clean):
    print(f"⚠️ Warning: Dropped {len(X_full) - len(X_clean)} rows with missing values.")
    # Align labels if rows were dropped
    if y_true_full is not None:
        y_true_full = y_true_full[X_clean.index]

# --- OPTIMIZATION 1: FEATURE SELECTION ---
CLUSTERING_FEATURES = [
    'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 
    'alamine_aminotransferase', 'aspartate_aminotransferase', 
    'total_protiens', 'albumin', 'albumin_and_globulin_ratio'
]
X_biomarkers = X_clean[CLUSTERING_FEATURES]

# --- OPTIMIZATION 2: OUTLIER REMOVAL (ISOLATION FOREST) ---
print("\n🧹 Running Isolation Forest to remove outliers...")
iso = IsolationForest(contamination=0.1, random_state=RANDOM_STATE) 
outlier_preds = iso.fit_predict(X_biomarkers)

# Keep only inliers (label == 1)
X_filtered = X_biomarkers[outlier_preds == 1]
X_filtered_full = X_clean[outlier_preds == 1] # For report
if y_true_full is not None:
    y_filtered = y_true_full[outlier_preds == 1]
else:
    y_filtered = None

print(f"✅ Removed {np.sum(outlier_preds == -1)} outliers. Remaining patients: {len(X_filtered)}")

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_filtered)

# --- OPTIMIZATION 3: UMAP PROJECTION ---
print("\n🌌 Running UMAP Manifold Projection...")
umap_reducer = UMAP(
    n_neighbors=30,      
    min_dist=0.0,        
    n_components=2,      
    random_state=RANDOM_STATE
)
X_umap = umap_reducer.fit_transform(X_scaled)
print("✅ UMAP projection complete.")

# =========================================================
# 2. PSO FOR OPTIMAL CLUSTER CENTROIDS
# =========================================================
print("\n🚀 Initiating Particle Swarm Optimization (PSO)...")

# Objective: Minimize Inertia in UMAP space
def kmeans_pso_objective(particles, data, n_clusters):
    n_particles = particles.shape[0]
    n_features = data.shape[1]
    losses = []
    for i in range(n_particles):
        centroids = particles[i].reshape(n_clusters, n_features)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        min_dists = np.min(distances, axis=1)
        losses.append(np.sum(min_dists ** 2))
    return np.array(losses)

N_CLUSTERS = 2 
N_FEATURES = X_umap.shape[1] 
DIMENSIONS = N_CLUSTERS * N_FEATURES
PARTICLES = 15
ITERATIONS = 100

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=PARTICLES, dimensions=DIMENSIONS, options=options)

cost, pos = optimizer.optimize(
    lambda p: kmeans_pso_objective(p, X_umap, N_CLUSTERS), 
    iters=ITERATIONS,
    verbose=True
)

best_centroids_umap = pos.reshape(N_CLUSTERS, N_FEATURES)
print("✅ PSO Optimization Complete.")

# =========================================================
# 3. FINAL CLUSTERING ASSIGNMENT
# =========================================================
final_kmeans = KMeans(n_clusters=N_CLUSTERS, init=best_centroids_umap, n_init=1, random_state=RANDOM_STATE)
clusters = final_kmeans.fit_predict(X_umap)

# Re-attach to data
X_final_report = X_filtered_full.copy()
X_final_report["Cluster"] = clusters

# Order by severity
cluster_severity = X_final_report.groupby("Cluster")['total_bilirubin'].mean().sort_values()
severity_map = {old: new for new, old in enumerate(cluster_severity.index)}
X_final_report["Cluster"] = X_final_report["Cluster"].map(severity_map)
clusters = X_final_report["Cluster"].values 

# =========================================================
# 4. REPORT GENERATION
# =========================================================
report_lines = []
report_lines.append("="*60)
report_lines.append("FINAL CLINICAL ANALYSIS REPORT (Manifold Optimized)")
report_lines.append("="*60)

# A. Metrics
report_lines.append("\n1. MODEL PERFORMANCE METRICS")

# Silhouette on UMAP space
sil_score = silhouette_score(X_umap, clusters)
report_lines.append(f"   - Silhouette Score: {sil_score*100:.2f}%")

ari = 0
nmi = 0
if y_filtered is not None:
    ari = adjusted_rand_score(y_filtered, clusters)
    nmi = normalized_mutual_info_score(y_filtered, clusters)
    report_lines.append(f"   - Adjusted Rand Index (ARI): {ari*100:.2f}%")
    report_lines.append(f"   - Normalized Mutual Info (NMI): {nmi*100:.2f}%")

# --- NEW: TOTAL COMPOSITE SCORE CALCULATION ---
# Calculate the mean of normalized metrics
norm_sil = max(0, sil_score)
norm_ari = max(0, ari)
norm_nmi = nmi # NMI is already 0-1

# If we have labels, average all 3. If not, just Silhouette.
if y_filtered is not None:
    total_score = (norm_sil + norm_ari + norm_nmi) / 3
else:
    total_score = norm_sil

report_lines.append(f"\n   🏆 TOTAL CLUSTER VALIDATION SCORE: {total_score * 100:.2f}/100")
report_lines.append(f"      (Composite of available metrics)")

# B. Biomarker Stats
report_lines.append("\n2. BIOMARKER SIGNIFICANCE")
report_lines.append(f"   {'Feature':<25} | {'P-Value':<12} | {'Significance'}")
report_lines.append("-" * 75)

for feature in CLUSTERING_FEATURES:
    groups = [X_final_report[X_final_report["Cluster"] == k][feature].values for k in range(N_CLUSTERS)]
    try:
        stat, p_val = kruskal(*groups)
        sig = "Significant" if p_val < 0.05 else "Not Significant"
        report_lines.append(f"   {feature:<25} | {p_val:.4f}       | {sig}")
    except ValueError: pass

print("\n".join(report_lines))

# Save Report
with open(os.path.join(OUTPUT_DIR, "final_report_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# Save Data
X_final_report.to_csv(os.path.join(OUTPUT_DIR, "final_clustered_patients.csv"), index=False)

# =========================================================
# 5. SAVE ARTIFACTS FOR PREDICT.PY
# =========================================================
print("\n💾 Saving artifacts...")
joblib.dump(scaler, os.path.join(MODEL_DIR, "unsupervised_scaler.pkl"))
joblib.dump(umap_reducer, os.path.join(MODEL_DIR, "umap_reducer.pkl"))

# Save Centroids (UMAP Space)
final_centroids_umap = []
for k in range(N_CLUSTERS):
    final_centroids_umap.append(X_umap[clusters == k].mean(axis=0))
np.save(os.path.join(MODEL_DIR, "pso_kmeans_centroids.npy"), np.array(final_centroids_umap))

print("✅ Artifacts saved.")