knn_scores = []

for k in range (1,40):
  knn_classifier = KNeighborsClassifier(n_neighbors = k)
  encoder = OneHotEncoder()
  knn_classifier.fit(X_train, y_train)
  knn_scores.append(knn_classifier.score(X_test, y_test))
print(f'best choice of k:{np.argmax(knn_scores)+1}')

k=8
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
print(f'accuracy: {np.sum(y_pred==y_test)/len(y_test)}')
