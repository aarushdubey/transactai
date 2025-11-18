import requests
import streamlit as st
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="TransactAI Demo", layout="centered")

st.title("TransactAI – Transaction Categorisation Demo")

description = st.text_input("Transaction description", "STARBUCKS #0421 MUMBAI IN")

if st.button("Predict"):
    resp = requests.post(f"{API_URL}/predict", json={"description": description})
    if resp.status_code == 200:
        data = resp.json()
        st.success(f"Predicted Category: {data['category']}")
        st.write(f"Confidence: {data['confidence']:.2f}")
        st.write("Top contributing n-grams:")
        st.code(", ".join(data["top_features"]))
    else:
        st.error("Error from API")

st.markdown("---")
st.subheader("Feedback (Human-in-the-loop)")
with st.form("feedback_form"):
    desc_fb = st.text_input("Description", description)
    predicted_category = st.text_input("Predicted category (copy from above)")
    predicted_confidence = st.number_input("Predicted confidence", 0.0, 1.0, 0.0, 0.01)
    corrected_category = st.text_input("Corrected category")
    comment = st.text_input("Comment (optional)")
    submitted = st.form_submit_button("Submit feedback")

    if submitted:
        payload = {
            "description": desc_fb,
            "predicted_category": predicted_category,
            "predicted_confidence": predicted_confidence,
            "corrected_category": corrected_category,
            "user_comment": comment,
        }
        r = requests.post(f"{API_URL}/feedback", json=payload)
        if r.status_code == 200:
            st.success("Feedback saved. Thank you!")
        else:
            st.error("Error saving feedback.")
