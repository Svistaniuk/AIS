import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pd.read_csv(url, names=names)

print("ВИВЧЕННЯ ДАНИХ")
print("Розмір даних:", dataset.shape)
print("\nПерші 20 рядків:\n", dataset.head(20))
print("\nСтатистичне зведення:\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())

print("\nВІЗУАЛІЗАЦІЯ ДАНИХ")
dataset.plot(kind='box', subplots=True, layout=(2,2), figsize=(8,6), sharex=False, sharey=False)
plt.suptitle('Діаграма розмаху атрибутів')
plt.tight_layout()
plt.show()

dataset.hist(figsize=(8,6))
plt.suptitle('Гістограма розподілу атрибутів')
plt.tight_layout()
plt.show()

scatter_matrix(dataset, figsize=(10,10))
plt.suptitle('Матриця діаграм розсіювання')
plt.tight_layout()
plt.show()

print("\nРОЗДІЛЕННЯ НА НАВЧАЛЬНИЙ ТА ТЕСТОВИЙ НАБОРИ")
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Навчальний набір: {X_train.shape[0]} зразків")
print(f"Тестовий набір: {X_test.shape[0]} зразків")

print("\nПОРІВНЯННЯ АЛГОРИТМІВ КЛАСИФІКАЦІЇ")
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print("\nТочність моделей (10-fold Stratified Cross-Validation):")
print("-" * 50)

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name:6s}: {cv_results.mean():.4f} (±{cv_results.std():.4f})")

plt.figure(figsize=(10,6))
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.ylabel('Точність (Accuracy)')
plt.xlabel('Алгоритм')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_idx = np.argmax([r.mean() for r in results])
best_name = names[best_idx]
best_model = models[best_idx][1]
print(f"\nНайкраща модель за результатами CV: {best_name}")
print(f"   Середня точність: {results[best_idx].mean():.4f}")

print("\nПРОГНОЗУВАННЯ НА ТЕСТОВОМУ НАБОРІ")
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

print(f"\nТочність на тестовому наборі: {accuracy_score(y_test, predictions):.4f}")
print("\nМатриця помилок:")
print(confusion_matrix(y_test, predictions))
print("\nЗвіт класифікації:")
print(classification_report(y_test, predictions))

print("\nПРОГНОЗ ДЛЯ НОВОЇ КВІТКИ")
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print(f"Вхідні дані: sepal-length={X_new[0,0]}, sepal-width={X_new[0,1]}, "
      f"petal-length={X_new[0,2]}, petal-width={X_new[0,3]}")
prediction = best_model.predict(X_new)
print(f"Прогнозований сорт ірису: {prediction[0]}")