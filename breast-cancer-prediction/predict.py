import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train the model
clf = LogisticRegression(max_iter=5000)
clf.fit(X, y)

# Sidebar for user input features
def user_input_features():
    # Adjust sliders to match the number of features (30 in this case)
    feature_input = {}
    for feature in X.columns:
        feature_input[feature] = st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

    features = pd.DataFrame(feature_input, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

# Make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display prediction results
st.subheader('Prediction')
st.write(data.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=data.target_names))
