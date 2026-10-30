import os
import joblib
import pandas as pd
import streamlit as st

from train_model import FEATURES, train

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.caption("Neural Network (MLP: 64 → 32 hidden units) trained on sample data")

MODEL_PATH = "house_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        train()  # train automatically if model file is missing
    return joblib.load(MODEL_PATH)


model = load_model()

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", 500, 4000, 1500, step=50)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
with col2:
    age = st.slider("Age of house (years)", 0, 40, 5)
    location = st.slider("Location score (1 = poor, 10 = prime)", 1, 10, 6)

if st.button("Predict Price", type="primary"):
    values = pd.DataFrame([[area, bedrooms, bathrooms, age, location]], columns=FEATURES)
    price = model.predict(values)[0]
    st.success(f"Estimated price: ₹ {price:,.2f} lakhs")

with st.expander("Network architecture"):
    mlp = model[-1]
    sizes = [mlp.n_features_in_] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]
    st.write("Layers: " + " → ".join(map(str, sizes)))
    st.write(f"Activation: {mlp.activation} | Epochs trained: {mlp.n_iter_}")
    st.write(f"Parameters: {sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_):,}")
