import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

n = len(x)

print("ІНТЕРПОЛЯЦІЯ ФУНКЦІЇ")
print(f"\nВхідні дані:")
print(f"x = {x}")
print(f"y = {y}")
print(f"Кількість точок: n = {n}")

print("Заповнення матриці Вандермонда")


X = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        X[i, j] = x[i] ** j

print("\nМатриця X:")
print(X)

det_X = np.linalg.det(X)
print(f"\nВизначник матриці: Δ = {det_X:.6e}")

print("Отримання коефіцієнтів полінома")


A = np.linalg.solve(X, y)

print("\nКоефіцієнти a0, a1, a2, a3, a4:")
for i, coef in enumerate(A):
    print(f"a{i} = {coef:.6f}")

print("Рівняння інтерполяційного полінома")


def polynomial(x_val, coeffs):
    """P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4"""
    result = 0
    for i, a in enumerate(coeffs):
        result += a * (x_val ** i)
    return result

equation = f"P(x) = {A[0]:.4f}"
for i in range(1, n):
    if A[i] >= 0:
        equation += f" + {A[i]:.4f}·x^{i}"
    else:
        equation += f" - {abs(A[i]):.4f}·x^{i}"
print(equation)

print("\nПеревірка у вузлах інтерполяції:")
print(f"{'i':<3} {'x':<8} {'y (задано)':<12} {'P(x)':<12} {'Похибка':<12}")
print("-" * 55)
for i in range(n):
    p_val = polynomial(x[i], A)
    error = abs(y[i] - p_val)
    print(f"{i:<3} {x[i]:<8} {y[i]:<12} {p_val:<12.6f} {error:<12.2e}")


plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='red', s=120, label='Вузли інтерполяції', 
            zorder=5, edgecolors='darkred', linewidth=2)

x_smooth = np.linspace(x.min() - 0.05, x.max() + 0.05, 300)
y_smooth = np.array([polynomial(xi, A) for xi in x_smooth])
plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, 
         label='Інтерполяційний поліном P(x)')

x_interp = np.array([0.2, 0.5])
y_interp = np.array([polynomial(xi, A) for xi in x_interp])
plt.scatter(x_interp, y_interp, color='green', s=100, marker='s', 
            label='Точки x=0.2 і x=0.5', zorder=5)

for xi, yi in zip(x_interp, y_interp):
    plt.text(xi, yi + 0.15, f'({xi}, {yi:.2f})', 
             ha='center', fontsize=9, color='green')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Інтерполяція поліномом 4-го степеня', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("Значення у проміжних точках")


x_interp_points = [0.2, 0.5]
print("\nЗначення інтерполяційного полінома:")
for x_val in x_interp_points:
    y_val = polynomial(x_val, A)
    print(f"P({x_val}) = {y_val:.6f}")