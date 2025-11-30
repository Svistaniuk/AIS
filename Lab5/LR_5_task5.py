import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

input_file = 'traffic_data.txt'
data = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')
        data.append(items)

data = np.array(data)

label_encoders = []
X_encoded = np.empty(data.shape, dtype=int)

for i in range(data.shape[1]):
    try:
        X_encoded[:, i] = data[:, i].astype(int)
    except ValueError:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1]  
y = X_encoded[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

clf = ExtraTreesClassifier(
    n_estimators=200,   
    max_depth=15,        
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Accuracy on test set: {accuracy:.2f}")

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_encoded = []

le_index = 0
for i, item in enumerate(test_datapoint):
    try:
        test_encoded.append(int(item))
    except ValueError:
        test_encoded.append(int(label_encoders[le_index].transform([item])[0]))
        le_index += 1

test_encoded = np.array(test_encoded).reshape(1, -1)
predicted = clf.predict(test_encoded)[0]
print(f"Predicted traffic intensity: {predicted}")