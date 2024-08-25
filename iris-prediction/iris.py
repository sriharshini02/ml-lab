import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
st.write(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

