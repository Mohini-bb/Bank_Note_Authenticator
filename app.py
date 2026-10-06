import streamlit as st
import pickle
import numpy as np

# Load model
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.title("Bank Authenticator 💵")

st.markdown(
    "<div style='background-color:#FF6347;padding:20px;border-radius:10px'>"
    "<h2 style='color:white;text-align:center'>Streamlit Bank Authenticator ML App</h2>"
    "</div><br>", unsafe_allow_html=True
)

variance = st.text_input("Variance", "2")
skewness = st.text_input("Skewness", "3")
curtosis = st.text_input("Curtosis", "1")
entropy = st.text_input("Entropy", "0.5")

if st.button("Predict"):
    try:
        data = np.array([[float(variance), float(skewness), float(curtosis), float(entropy)]])
        pred = classifier.predict(data)

        if pred[0] == 0:
            st.success("✅ REAL BANK NOTE! (Genuine)")
            st.balloons()
        else:
            st.error("❌ FAKE BANK NOTE! (Forged)")
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("About"):
    st.info("This app predicts if a bank note is Real or Fake using Random Forest (98.79% accuracy)")