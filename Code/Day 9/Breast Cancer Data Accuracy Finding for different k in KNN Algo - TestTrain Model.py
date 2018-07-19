from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

k_range = range(1,30)
scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	model = knn.fit(train, train_labels)
	preds = knn.predict(test)
	scores.append(accuracy_score(test_labels, preds))
	print(accuracy_score(test_labels, preds))

plt.plot(k_range,scores)
plt.xlabel("Value of k for KNN")
plt.ylabel("Testing Accuracy")
plt.show()
