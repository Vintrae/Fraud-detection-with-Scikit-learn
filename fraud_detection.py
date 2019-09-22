from sklearn import linear_model, preprocessing, metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Import dataset and perform data preprocessing.

data = pd.read_csv('/content/drive/My Drive/Notebooks/Fraud_detection/data/train_transaction.csv', low_memory=False)

# Visualise the data.
plt.rcParams['text.color'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'

fraud = data.loc[data['Class'] == 1]
legit = data.loc[data['Class'] == 0]
ax = fraud.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
legit.plot.scatter(x='Amount', y='Class', color='Blue', label='Legitimate', ax=ax)
plt.show

# Drop rows with missing values.
data.dropna(axis=0, inplace=True)

# Select features (Amount and 'V1', ..., 'V28' features) and label (Class)
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
label = 'Class'

# Create variables containing feature data and label data.
feature_data = data[features]
label_data = data[label]

# Normalise feature data.
scaler = preprocessing.StandardScaler()
feature_data = scaler.fit_transform(feature_data)

#Split into training and testing data.

# Create index for the split.
split_index = int(data.shape[0] * 0.8)

# Create training sets.
x_train = feature_data[:split_index]
y_train = label_data[:split_index]

# Create testing sets.
x_test = feature_data[split_index:]
y_test = label_data[split_index:]

#Create Logistic Regression model.

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# And finally: show the results
print(metrics.classification_report(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print()
