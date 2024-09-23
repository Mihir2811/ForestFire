import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('forestfires.csv')

data.head(5)

data.info()

data.describe()

print(data.columns)

sns.scatterplot(x='Area', y='Temperature ', data=data)

sns.scatterplot(x='Area', y='Relative humidity', data=data)

index = ['Fine Fuel Moisture Code',' Duff Moisture Code', 'Drought Code', 'Initial Spread Index','Temperature ', 'Relative humidity', 'Wind', 'Rain',]

for i in index:
  sns.scatterplot(x='Area', y=i, data=data)
  plt.show()

data['Area'].astype('float').unique()

data

data['Fire_category_risk'] = pd.cut(data['Area'], bins = [0, 5, float('inf')], labels=['low','high'])

data['Fire_category_risk']

data['Fire_category_risk'].isnull().sum()

data['Fire_category_risk'].unique()

data['Fire_category_risk'].info()

print(data.columns)

X = data.drop(['Area','Day', 'Month', 'Fire_category_risk'],axis=1)
y = data['Fire_category_risk']

#Encoding Target Variable
le = LabelEncoder()
y = le.fit_transform(y)

y

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

# Standardization

scaler = StandardScaler()

# Fit and transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the test data
X_test = scaler.transform(X_test)

# Support Vector Machine

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_score = accuracy_score(y_test, svm_pred)
print("SVM Accuracy: ", svm_score)

report_svm = classification_report(y_test, svm_pred)
print("Classification Report (SVM):\n", report_svm)

# Gaussia navie bayes

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_score = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_score)

report_nb = classification_report(y_test, nb_pred)
print("Classification Report (Naive Bayes):\n", report_nb)

# Decision Tree

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

tree_score = accuracy_score(y_test, tree_pred)
print("Accuracy of Decision Tree Classifier: ", tree_score)

tree_report = classification_report(y_test, tree_pred)
print("Classification Report (Decision Tree):\n", tree_report)

# Random Forest Classifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_model_pred = rf_model.predict(X_test)

rf_score = accuracy_score(y_test, rf_model_pred)
print("Accuracy of Random Forest Classifier: ", rf_score)

rf_report = classification_report(y_test, rf_model_pred)
print("Classification Report (Random Forest):\n", rf_report)

# Logistic Regression

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

logistic_model = logistic.fit(X_train, y_train)
logistic_model_pred = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_model_pred)
print("Accuracy of Logistic Regression: ", logistic_accuracy)

# Visualization

# Visualize Model Performance for Decision Tree and Random Forest
models = ['Decision Tree', 'Random Forest']
accuracies = [tree_score, rf_score]

plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Decision Tree vs Random Forest')
plt.ylim(0, 1)  # Adjust ylim for better visualization
#plt.show()

# Visualize Model Performance for SVM and Naive Bayes
models = ['SVM', 'Naive Bayes']
accuracies = [svm_score, nb_score]

plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison: SVM vs Naive Bayes')
plt.ylim(0, 1)  # Adjust ylim for better visualization
#plt.show()

# Visualize Model Performance
models = ['SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest']
accuracies = [svm_score, nb_score, tree_score, rf_score]

plt.bar(models, accuracies, color=('purple','red','blue','green'))
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)  # Adjust ylim for better visualization
#plt.show()

print(data.columns)

# Pickle the File

import pickle


filename = 'rf_model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))

# Prediction

# prompt: predict by our own values using random forest classifier

# Create a dictionary with your own input values for the features
pred_data = {'X': 7, 'Y': 5,  'Fine Fuel Moisture Code': 92.5,  ' Duff Moisture Code': 121.1,  'Drought Code': 674.4,  'Initial Spread Index': 8.6,  'Temperature ': 25.1,  'Relative humidity': 27,'Wind': 4, 'Rain': 0.0 }

# Create a DataFrame from the new data
new_data = pd.DataFrame([pred_data])
print(new_data)

# Scale the new data using the same scaler used for training
new_scaled = scaler.transform(new_data)
print(new_scaled)

# Make a prediction using the Random Forest model
prediction = rf_model.predict(new_scaled)
prediction2 = svm_model.predict(new_scaled)
prediction3 = nb_model.predict(new_scaled)
prediction4 = logistic_model.predict(new_scaled)

print('-----------------------------------------------')

# Print the prediction (0 for low risk, 1 for high risk)
print("Prediction 1 using Random Forest Classifier :", prediction)

if (prediction == 0):
  print("Low risk")
elif (1>=prediction>0):
  print("High risk")
else:
  print("Very High Risk")

print('-----------------------------------------------')


print("prediction 2 using SVM:", prediction2)

if (prediction2 == 0):
  print("Low risk")
elif (1>=prediction2>0):
  print("High risk")
else:
  print("Very High Risk")

print('-----------------------------------------------')

print("prediction 3 using Naive Bayes:", prediction3)

if (prediction3 == 0):
  print("Low risk")
elif (1>=prediction3>0):
  print("High risk")
else:
  print("Very High Risk")

print('-----------------------------------------------')

print("prediction 4 using Logistic Regression:", prediction4)

if (prediction4 == 0):
  print("Low risk")
elif (1>=prediction4>0):
  print("High risk")
else:
  print("Very High Risk")

print('-----------------------------------------------')

#Thank You for Going Through.