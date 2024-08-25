import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = load_breast_cancer()

selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = pd.DataFrame(data.data, columns=data.feature_names)[selected_features]
y = data.target

clf = LogisticRegression(max_iter=5000)
clf.fit(X, y)

def user_input_features():
    feature_input = {}
    for feature in selected_features:
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
