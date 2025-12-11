import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

try:
    X = np.loadtxt('data_clustering.txt', delimiter=',')
except OSError:
    print("Файл data_clustering.txt не знайдено.")
    exit()

num_clusters = 5

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Вхідні дані')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(X)

step_size = 0.01
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)

cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='o', s=210, linewidths=4, color='black',
            zorder=12, facecolors='black')

plt.title('Межі кластерів')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print("=" * 40)
print("РЕЗУЛЬТАТИ ОЦІНКИ ЯКОСТІ КЛАСТЕРИЗАЦІЇ")
print("=" * 40)

silhouette_avg = metrics.silhouette_score(X, kmeans.labels_)
print(f"Коефіцієнт силуету (Silhouette Score): {silhouette_avg:.4f}")

ch_score = metrics.calinski_harabasz_score(X, kmeans.labels_)
print(f"Індекс Калінскі-Харабаса: {ch_score:.4f}")

db_score = metrics.davies_bouldin_score(X, kmeans.labels_)
print(f"Індекс Девіса-Болдіна: {db_score:.4f}")
print("=" * 40)