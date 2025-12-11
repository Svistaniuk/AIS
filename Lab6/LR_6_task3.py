import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

encoders = {}
for col in ['Outlook', 'Humidity', 'Wind', 'Play']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[['Outlook', 'Humidity', 'Wind']]
y = df['Play']

model = GaussianNB()
model.fit(X, y)

var_outlook = 'Overcast'
var_humidity = 'High'
var_wind = 'Strong'

input_data = [
    encoders['Outlook'].transform([var_outlook])[0],
    encoders['Humidity'].transform([var_humidity])[0],
    encoders['Wind'].transform([var_wind])[0]
]

prediction_idx = model.predict([input_data])[0]
prediction_label = encoders['Play'].inverse_transform([prediction_idx])[0]
probabilities = model.predict_proba([input_data])[0]

print(f"Вхідні умови (Варіант 22): {var_outlook}, {var_humidity}, {var_wind}")
print("-" * 40)
print(f"Прогнозоване рішення: {prediction_label}")
print(f"Ймовірність 'No': {probabilities[0]:.4f}")
print(f"Ймовірність 'Yes': {probabilities[1]:.4f}")

if probabilities[1] > probabilities[0]:
    print("\nВисновок: Модель прогнозує, що гра відбудеться.")
else:
    print("\nВисновок: Модель прогнозує, що гра НЕ відбудеться.")