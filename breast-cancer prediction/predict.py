import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

# Sidebar for user input features
def user_input_features():
    mean_radius = st.sidebar.slider('Mean Radius', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    mean_texture = st.sidebar.slider('Mean Texture', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    mean_perimeter = st.sidebar.slider('Mean Perimeter', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    mean_area = st.sidebar.slider('Mean Area', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    mean_smoothness = st.sidebar.slider('Mean Smoothness', float(X[:,4].min()), float(X[:,4].max()), float(X[:,4].mean()))

    data = {
        'mean_radius': mean_radius,
        'mean_texture': mean_texture,
        'mean_perimeter': mean_perimeter,
        'mean_area': mean_area,
        'mean_smoothness': mean_smoothness
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Display user input features
st.subheader('User Input Features')
st.write(df)

# Make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display prediction results
st.subheader('Prediction')
st.write('Malignant' if prediction[0] == 0 else 'Benign')

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=data.target_names))

# Model accuracy
st.subheader('Model Accuracy')
st.write(clf.score(X_test, y_test))
