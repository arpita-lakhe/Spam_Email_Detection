import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# App Title
st.title("📧 Spam Email Detection App")
st.write("Enter an email message below to check whether it is **Spam** or **Not Spam**.")

# Text input
email_text = st.text_area(
    "Email Content",
    height=180,
    placeholder="Type or paste the email content here..."
)

# Prediction button
if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email content.")
    else:
        # Transform input
        transformed_text = vectorizer.transform([email_text])

      

        # Predict
        prediction = model.predict(transformed_text)[0]

        # Output result
        if prediction == 1:
            st.error("🚨 This email is classified as **SPAM**")
        else:
            st.success("✅ This email is classified as **NOT SPAM**")

# Footer
st.markdown("---")
st.markdown("Developed using **Python, Scikit-learn & Streamlit**")
