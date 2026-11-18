import streamlit as st
import joblib

st.set_page_config(
    page_title="Sarcasm Detection",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("models/linear_svm.pkl")

model = load_model()

st.title("🔍 Sarcasm Detection")
st.write(
    "An NLP application using TF-IDF and "
    "Linear Support Vector Machine."
)

st.info(
    "This application demonstrates a research model. "
    "Its training dataset has a known publisher-label bias."
)

headline = st.text_area(
    "Enter a news headline:",
    placeholder="Type or paste a headline here..."
)

if st.button("Predict Sarcasm", type="primary"):

    if not headline.strip():
        st.warning("Please enter a headline first.")

    else:
        prediction = model.predict([headline])[0]
        decision_score = model.decision_function([headline])[0]

        st.subheader("Prediction")

        if prediction == 1:
            st.success("Sarcasm detected")
        else:
            st.info("No sarcasm detected")

        st.caption(
            f"Predicted class: {prediction} "
            "(1 = Sarcasm, 0 = No Sarcasm)"
        )

        st.write(
            f"**SVM decision score:** {decision_score:.4f}"
        )

        st.caption(
            "The decision score is not a probability or "
            "a calibrated confidence percentage."
        )

st.divider()

st.caption(
    "Model: Linear SVM | Feature extraction: TF-IDF | "
    "Research prototype"
)
