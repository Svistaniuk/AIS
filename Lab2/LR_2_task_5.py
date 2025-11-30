import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

print("Класифікація Ridge\n")

iris = load_iris()
X, y = iris.data, iris.target

print("Дані завантажено:")
print(f"  Кількість зразків: {X.shape[0]}")
print(f"  Кількість ознак: {X.shape[1]}")
print(f"  Класи: {iris.target_names}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(f"Навчальний набір: {X_train.shape[0]} зразків")
print(f"Тестовий набір: {X_test.shape[0]} зразків\n")

print("Налаштування класифікатора Ridge:")
print("  - tol=1e-2 (допуск для критерію зупинки)")
print("  - solver='sag' (стохастичний градієнтний спуск)\n")

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

print("Навчання завершено.\n")

y_pred = clf.predict(X_test)

print("ПОКАЗНИКИ ЯКОСТІ КЛАСИФІКАЦІЇ\n")

accuracy = np.round(metrics.accuracy_score(y_test, y_pred), 4)
precision = np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4)
recall = np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4)
f1 = np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4)
cohen_kappa = np.round(metrics.cohen_kappa_score(y_test, y_pred), 4)
matthews = np.round(metrics.matthews_corrcoef(y_test, y_pred), 4)

print(f"Accuracy (Точність):           {accuracy}")
print(f"Precision (Точність):          {precision}")
print(f"Recall (Повнота):              {recall}")
print(f"F1 Score (F1-міра):            {f1}")
print(f"Cohen Kappa Score:             {cohen_kappa}")
print(f"Matthews Corrcoef:             {matthews}\n")

print("ЗВІТ ПРО КЛАСИФІКАЦІЮ\n")
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nМАТРИЦЯ ПОМИЛОК\n")
mat = confusion_matrix(y_test, y_pred)
print(mat)

plt.figure(figsize=(8, 6))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix - Ridge Classifier')
plt.tight_layout()

plt.savefig("Confusion.jpg", dpi=150, bbox_inches='tight')
print("\nГрафік збережено як 'Confusion.jpg'\n")


plt.show()