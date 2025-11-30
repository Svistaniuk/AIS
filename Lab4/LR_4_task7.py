import numpy as np
import matplotlib.pyplot as plt

X = np.array([-12, 29, 0, 4, 6, 8])
Y = np.array([-3, 0, 1, 2, 9, 5])

n = len(X)

print("МЕТОД НАЙМЕНШИХ КВАДРАТІВ - ВАРІАНТ 7")
print(f"\nВхідні дані:")
print(f"X = {X}")
print(f"Y = {Y}")

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X ** 2)

beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
beta_0 = (sum_y - beta_1 * sum_x) / n

print(f"\nПараметри лінійної регресії:")
print(f"β₀ (зсув) = {beta_0:.4f}")
print(f"β₁ (нахил) = {beta_1:.4f}")
print(f"\nРівняння: y = {beta_0:.4f} + {beta_1:.4f}·x")

Y_pred = beta_0 + beta_1 * X
S = np.sum((Y - Y_pred) ** 2)

print(f"\nСума квадратів помилок: S = {S:.4f}")

ss_tot = np.sum((Y - np.mean(Y)) ** 2)
ss_res = np.sum((Y - Y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Коефіцієнт детермінації: R² = {r_squared:.4f}")

print(f"\nТаблиця результатів:")
print(f"{'x':<8} {'y (факт)':<12} {'y (прогноз)':<12} {'Похибка':<10}")
print("-" * 45)
for i in range(n):
    error = Y[i] - Y_pred[i]
    print(f"{X[i]:<8} {Y[i]:<12} {Y_pred[i]:<12.4f} {error:<10.4f}")

# Побудова графіка
plt.figure(figsize=(10, 6))

plt.scatter(X, Y, color='red', s=100, label='Експериментальні точки', zorder=3)

x_line = np.linspace(X.min() - 2, X.max() + 2, 100)
y_line = beta_0 + beta_1 * x_line
plt.plot(x_line, y_line, 'b-', linewidth=2, 
         label=f'y = {beta_0:.2f} + {beta_1:.2f}·x')

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Лінійна регресія (Варіант 7)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()