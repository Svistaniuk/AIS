import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import ExtraTreesClassifier
from utilities import visualize_classifier 

input_file = 'data_random_forests.txt'
try:
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
except OSError:
    print(f"Помилка: Файл {input_file} не знайдено. Перевірте наявність.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

parameter_grid = [
    { 'n_estimators': [100],
      'max_depth': [2, 4, 7, 12, 16] 
    },
    { 'max_depth': [4], 
      'n_estimators': [25, 50, 100, 250] 
    }
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n" + "#"*40)
    print(f"##### Searching optimal parameters for {metric}")
    
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0), 
        parameter_grid, 
        cv=5, 
        scoring=metric
    )
    
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    for params, avg_score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(params, '-->', round(avg_score, 3))
        
    print("\nBest parameters:", classifier.best_params_)
    print("Best score:", round(classifier.best_score_, 3))

    y_pred = classifier.predict(X_test)
    print("\nPerformance report on test set with best parameters:\n")
    print(classification_report(y_test, y_pred))
    print("#"*40)

print("\nВізуалізація меж рішень з найкращими параметрами для останньої метрики...")
visualize_classifier(classifier.best_estimator_, X_test, y_test, 'Grid Search Optimal Model Boundaries')