import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
try:
    df = pd.read_csv(url)
    print("Дані завантажено з інтернету.")
except:
    df = pd.read_csv('renfe_small.csv')
    print("Дані завантажено з локального файлу.")

df = df.dropna(subset=['price', 'train_class', 'train_type', 'fare'])

df = df.sample(5000, random_state=42)

print(f"Кількість записів для аналізу: {len(df)}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['price'], kde=True, color='skyblue')
plt.title('Розподіл цін на квитки')
plt.xlabel('Ціна (€)')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
sns.boxplot(x='train_type', y='price', data=df)
plt.title('Ціна залежно від типу поїзда')
plt.xlabel('Тип поїзда')
plt.ylabel('Ціна (€)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

le_type = LabelEncoder()
le_class = LabelEncoder()

df['train_type_code'] = le_type.fit_transform(df['train_type'])
y = le_class.fit_transform(df['train_class'])

X = df[['price', 'train_type_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nТочність моделі: {accuracy:.2%}")
print("\nДетальний звіт класифікації:")
print(classification_report(y_test, y_pred, target_names=le_class.classes_))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_class.classes_,
            yticklabels=le_class.classes_)
plt.title('Матриця плутанини (Confusion Matrix)')
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений моделлю клас')
plt.show()