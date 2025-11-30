import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import urllib.request
import os

input_file = 'income_data.txt'

if not os.path.exists(input_file):
    print("Файл не знайдено. Завантажуємо дані з UCI Repository...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    try:
        urllib.request.urlretrieve(url, input_file)
        print(f"Дані успішно завантажено у файл '{input_file}'")
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        print("Будь ласка, завантажте файл вручну з:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
        exit()

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

print(f"\nЗавантаження даних з файлу '{input_file}'...")

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        
        data = line[:-1].split(', ')
        
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

print(f"Завантажено {count_class1} зразків класу '<=50K'")
print(f"Завантажено {count_class2} зразків класу '>50K'")
print(f"Загалом: {count_class1 + count_class2} зразків\n")

X = np.array(X)

print("Кодування категоріальних ознак...")
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
print("Кодування завершено\n")

print("Розбиття даних на навчальний (80%) та тестовий (20%) набори...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f"Навчальний набір: {len(X_train)} зразків")
print(f"Тестовий набір: {len(X_test)} зразків\n")

print("Навчання SVM-класифікатора...")
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=2000))
classifier.fit(X_train, y_train)
print("Навчання завершено\n")

print("Прогнозування на тестовому наборі...")
y_test_pred = classifier.predict(X_test)
accuracy = np.mean(y_test_pred == y_test)
print(f"Точність на тестовому наборі: {round(100*accuracy, 2)}%\n")

print("Обчислення F1-score з 3-fold перехресною перевіркою...")
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print(f"F1 score: {round(100*f1.mean(), 2)}%\n")

print("ТЕСТУВАННЯ НА НОВІЙ ТОЧЦІ ДАНИХ")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

print("\nВхідні дані:")
print(f"  Вік: {input_data[0]}")
print(f"  Тип зайнятості: {input_data[1]}")
print(f"  Освіта: {input_data[3]}")
print(f"  Сімейний стан: {input_data[5]}")
print(f"  Професія: {input_data[6]}")
print(f"  Стать: {input_data[9]}")
print(f"  Годин на тиждень: {input_data[12]}")

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array([input_data_encoded])
predicted_class = classifier.predict(input_data_encoded)
result = label_encoder[-1].inverse_transform([predicted_class[0]])[0]

print(f"ПРОГНОЗ: Річний дохід особи - {result}")
