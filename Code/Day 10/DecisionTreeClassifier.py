# Importing Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load dataset
data = load_breast_cancer()

# Organize the data
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
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Initialize our classifier
dtf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Train our classifier
model = dtf.fit(x_train, y_train)

# Make predictions
preds = dtf.predict(x_test)

# Evaluate accuracy
print("Accuracy:",round(accuracy_score(y_test, preds)*100,2),"%")

#Making Confusion Matrix
cm = confusion_matrix(y_test, preds)
print(cm)