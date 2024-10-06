import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
X = np.array([
    [56, 180, 80], [45, 170, 65], [60, 160, 70], [30, 175, 68], [50, 172, 72],
    [40, 165, 75], [55, 150, 60], [65, 140, 85], [48, 190, 77], [52, 155, 62],
    [58, 160, 68], [49, 175, 78], [33, 185, 72], [47, 167, 71], [44, 173, 66],
    [39, 170, 64], [60, 160, 70], [34, 176, 74], [55, 155, 67], [36, 162, 61],
    [50, 178, 76], [61, 159, 65], [40, 150, 72], [51, 145, 63], [62, 172, 70],
    [37, 160, 66], [53, 185, 73], [54, 176, 77], [59, 170, 71], [43, 169, 68]
])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.labels_
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f'K-Means Inertia: {kmeans_inertia}')
print(f'K-Means Silhouette Score: {kmeans_silhouette}')
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)
gmm_log_likelihood = gmm.score(X_scaled)
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
print(f'GMM Log Likelihood: {gmm_log_likelihood}')
print(f'GMM Silhouette Score: {gmm_silhouette}')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis')
plt.title('GMM Clustering')
plt.show()