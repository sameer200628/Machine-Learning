#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Dataset
data = pd.DataFrame({
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
})

print("Training Dataset\n")
print(data)

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def find_s(concepts, target):
    hypothesis = concepts[0].copy()

    for i in range(len(target)):
        if target[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != concepts[i][j]:
                    hypothesis[j] = '?'

    return hypothesis

hypothesis = find_s(concepts, target)

print("\nMost Specific Hypothesis:")
print(hypothesis)


# In[1]:


import pandas as pd

data = pd.DataFrame([
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
], columns=['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport'])

concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

specific = concepts[0].copy()
general = [["?" for _ in range(len(specific))] for _ in range(len(specific))]

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(specific)):
            if h[x] != specific[x]:
                specific[x] = "?"
                general[x][x] = "?"
    else:
        for x in range(len(specific)):
            if h[x] != specific[x]:
                general[x][x] = specific[x]
            else:
                general[x][x] = "?"

general = [g for g in general if g != ["?"] * len(specific)]

print("Specific Hypothesis:")
print(specific)

print("\nGeneral Hypothesis:")
for g in general:
    print(g)


# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
                    'Cool', 'Mild', 'Cool', 'Mild', 'Mild',
                    'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
             'Yes', 'No', 'Yes', 'Yes', 'Yes',
             'Yes', 'Yes', 'No']
})

# Separate features and target
X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']].copy()
y = data['Play'].copy()

# Encode categorical values
encoders = {}

for col in X.columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train ID3 Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# New sample to classify
sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

# Encode sample
for col in sample.columns:
    sample[col] = encoders[col].transform(sample[col])

# Predict
prediction = model.predict(sample)

print("Prediction:", target_encoder.inverse_transform(prediction)[0])


# In[5]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [
    [5.1,3.5,1.4,0.2],
    [4.9,3.0,1.4,0.2],
    [5.8,2.7,5.1,1.9],
    [6.0,2.2,5.0,1.5],
    [5.5,2.3,4.0,1.3],
    [6.5,2.8,4.6,1.5],
    [5.7,4.4,1.5,0.4],
    [6.7,3.1,4.7,1.5],
    [6.3,3.3,6.0,2.5]
]

y = [0,0,2,2,1,1,0,1,2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    solver='lbfgs',
    max_iter=1000,
    random_state=1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Predictions:", pred)
print("Accuracy:", accuracy_score(y_test, pred))


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset (Iris)
X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1],
    [7.3, 2.9, 6.3, 1.8],
    [6.7, 3.3, 5.7, 2.5],
    [7.2, 3.2, 6.0, 1.8]
]

# Target classes
# 0 = Setosa, 1 = Versicolor, 2 = Virginica
y = [0, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     2, 2, 2, 2, 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Predict test data
pred = model.predict(X_test)

# Display results
print("Predicted Classes:", pred)
print("Actual Classes:   ", y_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Predict a new sample
new_sample = [[5.9, 3.0, 5.1, 1.8]]
prediction = model.predict(new_sample)

species = ["Setosa", "Versicolor", "Virginica"]

print("\nNew Sample:", new_sample[0])
print("Predicted Species:", species[prediction[0]])


# In[ ]:




