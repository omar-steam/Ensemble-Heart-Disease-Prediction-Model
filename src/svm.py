m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))

# or this support vector machine technique

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform, randint

# Define the parameter grid
param_dist = {
    'C': uniform(loc=0, scale=10),  # Search range for C (you can adjust the scale based on your data)
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernels to be tried
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))  # Search range for gamma
}

# Create an SVC instance
svc = SVC()

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(
    svc,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',  # You can change the scoring metric based on your preference
    random_state=42,
    n_jobs=-1  # Use all available CPUs for parallel processing
)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters: ", random_search.best_params_)
print("Best Accuracy: ", random_search.best_score_)

# Get the best estimator from the search
best_svc = random_search.best_estimator_

# Use the best estimator to make predictions on the test set
best_svc_predicted = best_svc.predict(X_test)

# Evaluate the performance of the best estimator
best_svc_conf_matrix = confusion_matrix(y_test, best_svc_predicted)
best_svc_acc_score = accuracy_score(y_test, best_svc_predicted)

print("\n")
print("Confusion Matrix for Best SVC:")
print(best_svc_conf_matrix)
print("\n")
print("Accuracy of Best SVC:", best_svc_acc_score * 100, '\n')
print(classification_report(y_test, best_svc_predicted))
