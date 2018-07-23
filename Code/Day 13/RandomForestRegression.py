# Importing Libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#To ignore the warning messages
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Load dataset
data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:2].values
y = data.iloc[:, [2]].values

# Split our data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# Initialize our classifier
clf = RandomForestRegressor(n_estimators = 10, random_state = 0)

# Train our classifier
model = clf.fit(x_train, y_train)

# Make predictions
preds = clf.predict(6.5)
print(preds)