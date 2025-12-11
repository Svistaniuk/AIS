import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from itertools import cycle

def ensure_data_exists(filename):
    """Створює файл з даними, якщо він відсутній, щоб скрипт працював відразу."""
    if not os.path.exists(filename):
        print(f"Файл {filename} не знайдено. Генерую дані...")
        X, _ = make_blobs(n_samples=350, centers=5, cluster_std=0.8, random_state=42)
        np.savetxt(filename, X, delimiter=',')
        print("Файл успішно створено.")

def main():
    filename = 'data_clustering.txt'
    ensure_data_exists(filename)

    X = np.loadtxt(filename, delimiter=',')

    bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

    meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
    meanshift_model.fit(X)

    cluster_centers = meanshift_model.cluster_centers_
    labels = meanshift_model.labels_
    num_clusters = len(np.unique(labels))

    print("\nCenters of clusters:")
    print(cluster_centers)
    print(f"\nNumber of clusters in input data = {num_clusters}")

    plt.figure()
    markers = cycle('oxvsD')  
    
    for i, marker in zip(range(num_clusters), markers):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    marker=marker, color='black', s=50)
        
        center = cluster_centers[i]
        plt.plot(center[0], center[1], marker='o', markerfacecolor='black', 
                 markeredgecolor='black', markersize=15)

    plt.title('Кластери')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()