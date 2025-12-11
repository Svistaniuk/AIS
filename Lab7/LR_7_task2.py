import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
y_true = iris.target

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Центроїди')
axes[0].set_title('Результат кластеризації K-Means')
axes[0].set_xlabel('Довжина чашолистка (см)')
axes[0].set_ylabel('Ширина чашолистка (см)')
axes[0].legend()

axes[1].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
axes[1].set_title('Справжні класи (Ground Truth)')
axes[1].set_xlabel('Довжина чашолистка (см)')
axes[1].set_ylabel('Ширина чашолистка (см)')

plt.suptitle('Порівняння результатів K-Means зі справжніми даними Iris', fontsize=16)
plt.show()