import streamlit as st
import pickle

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detection App")
st.write("Enter an email message below to check whether it is Spam or Not Spam.")

email_text = st.text_area("Email Content", height=180)

if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("Please enter email content.")
    else:
        transformed_text = vectorizer.transform([email_text])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT SPAM")
