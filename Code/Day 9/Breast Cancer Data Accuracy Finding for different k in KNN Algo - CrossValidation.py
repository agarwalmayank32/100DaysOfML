from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 

import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at our data
print(label_names)
print('Class label = ', labels[0])
print(feature_names)
print(features[0])

k_range = range(1,20)
k_scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, features, labels, cv=10, scoring='accuracy')
	k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel("Value of k for KNN")
plt.ylabel("Testing Accuracy")
plt.show()
