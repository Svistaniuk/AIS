import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

FILE_PATH = "income_data.txt"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

try:
    data = pd.read_csv(FILE_PATH, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
    print(f"Дані успішно завантажено! Розмір набору: {data.shape}")
except FileNotFoundError:
    print(f"Помилка: файл '{FILE_PATH}' не знайдено.")
    exit()

data.dropna(inplace=True)

for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop("income", axis=1)
y = data["income"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nПісля обробки: {X_train.shape[0]} навчальних та {X_test.shape[0]} тестових прикладів")

models = [
    ("LR", LogisticRegression(solver='liblinear', random_state=42)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("CART", DecisionTreeClassifier(random_state=42)),
    ("NB", GaussianNB()),
    ("SVM", SVC(kernel='rbf', gamma='auto', random_state=42))
]

results = {}
print("\nПорівняння якості моделей (10-кратна стратифікована крос-валідація):\n")

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    results[name] = cv_scores
    print(f"{name:>4}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

plt.figure(figsize=(9,6))
plt.boxplot(results.values(), labels=results.keys())
plt.title("Порівняння алгоритмів класифікації (Income Data)")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.show()

best_model_name = max(results, key=lambda x: results[x].mean())
best_model_score = results[best_model_name].mean()

print(f"\nНайкраща модель за середньою точністю: {best_model_name} ({best_model_score:.4f})")

for name, model in models:
    if name == best_model_name:
        best_model = model
        break

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nОцінка на тестовій вибірці:")
print(f"Точність (accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nМатриця помилок:")
print(confusion_matrix(y_test, y_pred))
print("\nЗвіт про класифікацію:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
